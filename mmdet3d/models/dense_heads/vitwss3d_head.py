from mmdet.models.builder import HEADS
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import build_bbox_coder, multi_apply
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import ipdb
from mmdet3d.core import bbox

from mmdet3d.core.bbox.structures.base_box3d import BaseInstance3DBoxes
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.models.builder import build_loss
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D, BboxOverlapsNearest3D


@HEADS.register_module()
class ViTWSS3DHead(BaseModule):
    def __init__(
        self,
        num_classes,
        in_channels,
        bbox_coder,
        num_mlp_layers,
        mlp_channel,
        matcher_cfg: dict,
        train_cfg=None,
        test_cfg=None,
        sem_loss=None,
        cls_loss_type = 'cls', #cls or iou
        center_loss=None,
        center_res_type = 'direct',
        size_loss=None,
        dir_cls_loss=None,
        dir_res_loss=None,
        corner_loss=None,
        init_cfg=None,
        iou_loss = None

    ):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_mlp_layers = num_mlp_layers
        self.mlp_channel = mlp_channel

        # bbox_coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_dir_bins = self.bbox_coder.num_dir_bins
        # matcher used for gt_bboxes assignment
        matcher_type = matcher_cfg.get('type', 'HungaryMatcher')
        matcher_cfg.pop('type', None)
        if matcher_type == 'HungaryMatcher':
            self.matcher = _HungaryMatcher(**matcher_cfg)
        elif matcher_type == 'HungaryMatcherV2':
            self.matcher = _HungaryMatcherV2(**matcher_cfg)
        elif matcher_type == 'SequentialMatcher':
            self.matcher = _SequentialMatcher(**matcher_cfg)
        elif matcher_type == 'PointsInsideMatcher':
            self.matcher = _PointsInsideMatcher(**matcher_cfg)
        else:
            raise NotImplementedError
        # prediction head
        self.cls_loss_type = cls_loss_type

        if self.cls_loss_type == 'cls':
            self.class_embed = _MLP(self.in_channels, self.mlp_channel, self._get_cls_out_channels(), self.num_mlp_layers)
        elif self.cls_loss_type == 'iou':
            self.iou_embed = _MLP(self.in_channels, self.mlp_channel, 1, self.num_mlp_layers)
        # self.bbox_embed = _MLP(self.in_channels, self.mlp_channel, self._get_reg_out_channels(), self.num_mlp_layers)

        self.center_embed = _MLP(self.in_channels, self.mlp_channel, 3, self.num_mlp_layers)
        self.size_embed = _MLP(self.in_channels, self.mlp_channel, 3, self.num_mlp_layers)
        self.dir_embed = _MLP(self.in_channels, self.mlp_channel, self.num_dir_bins * 2, self.num_mlp_layers)


        # loss functions
        if self.cls_loss_type == 'cls':
            self.sem_loss = build_loss(sem_loss)
        else:
            self.sem_loss = build_loss(iou_loss)
        self.center_loss = build_loss(center_loss)
        self.size_loss = build_loss(size_loss)
        self.dir_cls_loss = build_loss(dir_cls_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.corner_loss = build_loss(corner_loss)
        self.iou_loss = build_loss(iou_loss)

        #loss config

        #center_res_config
        self.center_res_type = center_res_type

    def _get_cls_out_channels(self):
        return self.num_classes + 1

    def _get_reg_out_channels(self):
        # x, y, z, dx, dy, dz, (bin_cls) * num_dir_bins, (bin_reg) * num_dir_bins
        return 3 + 3 + self.num_dir_bins * 2

    def forward(self, backbone_outputs: dict):
        '''Forward pass

        Args:
            backbone_feats (torch.Tensor): the features extracted from backbone with shape of (B, #[DET], C)
        '''
        backbone_feats = backbone_outputs['features']
        det_token_center = backbone_outputs['det_token_center']
        
        B, num_det_tokens = backbone_feats.shape[0], backbone_feats.shape[1]

        if self.cls_loss_type == 'cls':
            cls_pred_logits = self.class_embed(backbone_feats)  # B, #[DET], (num_classes + 1)
        elif self.cls_loss_type == 'iou':
            cls_pred_logits = self.iou_embed(backbone_feats)  # B, #[DET], (num_classes + 1)
        # bbox_logits = self.bbox_embed(backbone_feats)  # B, #[DET], reg_out_channels

        #if self.cls_loss_type:
        center_logits = self.center_embed(backbone_feats)

        size_logits = self.size_embed(backbone_feats)
        dir_logits = self.dir_embed(backbone_feats)
        bbox_logits = torch.cat([center_logits, size_logits, dir_logits], dim=-1)

        # cls_pred_logits.register_hook(lambda grad: print('cls_pred', torch.norm(grad, 2)))
        # bbox_logits.register_hook(lambda grad: print('bbox_logits', torch.norm(grad, 2)))

        base_xyz = det_token_center
        if self.cls_loss_type == 'cls':
            bbox_results = self.bbox_coder.split_pred(
                cls_pred_logits.softmax(-1),
                bbox_logits.transpose(1, 2),
                base_xyz,
                center_res_type = self.center_res_type
            )
        else:
            bbox_results = self.bbox_coder.split_pred(
                cls_pred_logits,
                bbox_logits.transpose(1, 2),
                base_xyz,
                center_res_type = self.center_res_type
            )

        decoded_bbox = self.bbox_coder.decode(bbox_results, base_xyz=base_xyz, center_res_type=self.center_res_type)  # B, #[DET], 7

        # results:
        #       obj_scores
        #       center
        #       size
        #       dir_class
        #       dir_res_norm
        #       dir_res
        #       cls_pred_logits
        #       bbox3d_pred
        results = {
            'det_token_center': det_token_center,
            'cls_pred_logits': cls_pred_logits,
            'bbox3d_pred': decoded_bbox,  # gravity center
            #'iou_pred':iou_logits
        }


        results.update(bbox_results)
        return results

    @force_fp32(apply_to=('bbox_preds'))
    def loss(
        self,
        bbox_preds,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        return_targets=False,
    ):
        cls_pred_logits = bbox_preds['cls_pred_logits']
        obj_pred_scores = bbox_preds['obj_scores']
        center_pred = bbox_preds['center']  # TODO: gravity center?
        size_pred = bbox_preds['size']  # TODO: encoded size?
        dir_class_pred = bbox_preds['dir_class']
        dir_res_norm_pred = bbox_preds['dir_res_norm']
        det_token_center = bbox_preds['det_token_center']


        targets = self.get_targets(bbox_preds, gt_bboxes_3d, gt_labels_3d, img_metas, det_token_center)
        (label_targets, center_targets, size_targets, dir_class_targets, dir_res_targets,
        corner_targets, proposal_matched_mask, valid_gt_masks, iou_targets, bbox_targes) = targets

        losses = dict()

        # classification loss or iou loss
        if self.cls_loss_type == 'cls':
            obj_cls_loss = self.sem_loss(
                cls_pred_logits.reshape(-1, cls_pred_logits.shape[-1]),
                label_targets.flatten(),
            ).sum(-1)
            # obj_cls_loss = obj_cls_loss * proposal_matched_mask.flatten() * 5 + obj_cls_loss
            obj_cls_loss = torch.mean(obj_cls_loss)
            losses['obj_cls_loss'] = obj_cls_loss
        # cls_pred_logits.register_hook(lambda grad: print(torch.norm(grad, 2)))
        
        if valid_gt_masks.sum() > 0:

            # iou regression loss
            if self.cls_loss_type == 'iou':
                obj_cls_loss = self.sem_loss(
                    cls_pred_logits,
                    iou_targets.unsqueeze(2),
                ).sum(-1)
                # print(f'pred:{center_pred}\ttargets:{center_targets}')
                obj_cls_loss = obj_cls_loss * proposal_matched_mask
                obj_cls_loss = obj_cls_loss.sum() / valid_gt_masks.sum()  # mean
                losses['obj_cls_loss'] = obj_cls_loss
            #print(obj_cls_loss, iou_targets)
            # center regression loss
            obj_center_loss = self.center_loss(
                center_pred,  # TODO: gravity center?
                center_targets,  # gravity center
            ).sum(-1)
            # print(f'pred:{center_pred}\ttargets:{center_targets}')
            obj_center_loss = obj_center_loss * proposal_matched_mask
            obj_center_loss = obj_center_loss.sum() / valid_gt_masks.sum()  # mean

            # size regression loss
            obj_size_loss = self.size_loss(
                size_pred,
                size_targets  # encoded size
            ).sum(-1)
            obj_size_loss *= proposal_matched_mask
            obj_size_loss = obj_size_loss.sum() / valid_gt_masks.sum()  # mean

            # direction regression loss
            obj_dir_cls_loss = self.dir_cls_loss(
                dir_class_pred.reshape(-1, dir_class_pred.shape[-1]),
                dir_class_targets.flatten()
            )
            obj_dir_cls_loss *= proposal_matched_mask.flatten()
            obj_dir_cls_loss = obj_dir_cls_loss.sum() / valid_gt_masks.sum()  # mean

            dir_cls_targets_one_hot = torch.zeros_like(dir_res_norm_pred, dtype=torch.float32)
            dir_cls_targets_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
            dir_res_pred_for_gt = torch.sum(dir_res_norm_pred * dir_cls_targets_one_hot, -1)
            obj_dir_res_loss = self.dir_res_loss(
                dir_res_pred_for_gt,
                dir_res_targets
            )
            obj_dir_res_loss *= proposal_matched_mask
            obj_dir_res_loss = obj_dir_res_loss.sum() / valid_gt_masks.sum()  # mean

            # corner loss
            decoded_pred_bbox3d = bbox_preds['bbox3d_pred']  # TODO: gravity center?
            decoded_pred_bbox3d = decoded_pred_bbox3d.reshape(-1, decoded_pred_bbox3d.shape[-1])
            pred_bbox3d_corners = img_metas[0]['box_type_3d'](
                decoded_pred_bbox3d.clone(),
                box_dim=decoded_pred_bbox3d.shape[-1],
                with_yaw=self.bbox_coder.with_rot,
                origin=(0.5, 0.5, 0.5)  # that's to say, decode_pred_bbox3d should be gravity centered
            ).corners
            obj_corner_loss = self.corner_loss(
                pred_bbox3d_corners.reshape(decoded_pred_bbox3d.shape[0], -1),
                corner_targets.reshape(decoded_pred_bbox3d.shape[0], -1)
            ).sum(-1)
            obj_corner_loss *= proposal_matched_mask.flatten()
            obj_corner_loss = obj_corner_loss.sum() / valid_gt_masks.sum()  # mean

            #rotate_iou_loss
            obj_iou_loss = self.iou_loss(bbox_preds['bbox3d_pred'].reshape(-1,7), bbox_targes.reshape(-1,7))
            obj_iou_loss = obj_iou_loss * proposal_matched_mask.flatten()
            obj_iou_loss = obj_iou_loss.sum() / valid_gt_masks.sum() # mean

        else:
            obj_center_loss = torch.zeros(1, device=center_pred.device).squeeze()
            obj_size_loss = torch.zeros(1, device=center_pred.device).squeeze()
            obj_dir_cls_loss = torch.zeros(1, device=center_pred.device).squeeze()
            obj_dir_res_loss = torch.zeros(1, device=center_pred.device).squeeze()
            obj_corner_loss = torch.zeros(1, device=center_pred.device).squeeze()
            obj_iou_loss = torch.zeros(1, device=center_pred.device).squeeze()

        # calculate the classification accuracy
        pred_labels = cls_pred_logits.reshape(-1, cls_pred_logits.shape[-1]).argmax(-1)
        label_targets_flatten = label_targets.flatten()
        # print('pred', pred_labels)
        for label_id in range(cls_pred_logits.shape[-1]):
            cur_obj_mask = label_targets_flatten == label_id
            if cur_obj_mask.sum() > 0:
                cur_pred = pred_labels[cur_obj_mask]
                category_acc = (cur_pred == label_id).sum() / cur_obj_mask.sum()
            else:
                category_acc = torch.ones((1), device=cls_pred_logits.device) * 0.5
            category_acc.detach()
            losses[f'category_{label_id}_acc'] = category_acc

        losses['obj_center_loss'] = obj_center_loss
        losses['obj_size_loss'] = obj_size_loss
        losses['obj_dir_cls_loss'] = obj_dir_cls_loss
        losses['obj_dir_res_loss'] = obj_dir_res_loss
        losses['obj_corner_loss'] = obj_corner_loss
        losses['obj_iou_loss'] = obj_iou_loss

        if return_targets:
            losses['targets'] = targets
        return losses

    def get_targets(
        self,
        bbox_preds,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        det_token_center=None,
    ):
        '''calculate the prediction targets.

        Args:
            bbox_preds (dict): the dict of model prediction
            gt_bboxes_3d (list[obj: BaseInstanceBBox3d]): the ground truth bboxes
            gt_labels_3d (lsit[torch.Tensor]): the labels
            img_metas (list[dict]): the image meta info
        Return:
            tuple[torch.Tensor]: training targets
        '''
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            # ipdb.set_trace()
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                gt_fg_mask = gt_labels_3d[index] >= 0
                gt_bboxes_3d[index] = gt_bboxes_3d[index][gt_fg_mask]
                gt_labels_3d[index] = gt_labels_3d[index][gt_fg_mask]
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        
        # cls_pred_scores_list = [
        #     bbox_preds['obj_scores'][i] for i in range(len(gt_labels_3d))
        # ]
        cls_pred_logits_list = [
            bbox_preds['cls_pred_logits'][i] for i in range(len(gt_labels_3d))
        ]
        bbox3d_pred_list = [
            bbox_preds['bbox3d_pred'][i] for i in range(len(gt_labels_3d))
        ]
        if det_token_center is not None:
            det_token_center_list = [
                det_token_center[i] for i in range(len(gt_labels_3d))
            ]
        else:
            det_token_center_list = [None for _ in range(len(gt_labels_3d))]

        (
            label_targets_list, center_targets_list,
            size_targets_list, 
            dir_class_targets_list, dir_res_targets_list,
            corner_targets_list,
            proposal_matched_mask_list,
            iou_targets_list,
            bbox_targets_list
        ) = multi_apply(
            self.get_targets_single,
            cls_pred_logits_list,
            bbox3d_pred_list,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_num,
            img_metas,
            det_token_center_list,
            # max_gt_nums,
        )

        for index in range(len(gt_labels_3d)):
            pad_num = max(gt_num) - gt_labels_3d[index].shape[0]
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        label_targets = torch.stack(label_targets_list)
        center_targets = torch.stack(center_targets_list)
        size_targets = torch.stack(size_targets_list)
        dir_class_targets = torch.stack(dir_class_targets_list)
        dir_res_targets = torch.stack(dir_res_targets_list)
        corner_targets = torch.stack(corner_targets_list)
        proposal_matched_mask = torch.stack(proposal_matched_mask_list)
        valid_gt_masks = torch.stack(valid_gt_masks)
        iou_targets = torch.stack(iou_targets_list)
        bbox_targes = torch.stack(bbox_targets_list)

        return label_targets, \
                center_targets,\
                size_targets, \
                dir_class_targets, dir_res_targets, \
                corner_targets, \
                proposal_matched_mask, valid_gt_masks, \
                iou_targets, \
                bbox_targes

    def get_targets_single(
        self,
        cls_pred_logits: torch.Tensor,
        bbox3d_pred: torch.Tensor,
        gt_bboxes_3d: BaseInstance3DBoxes,
        gt_labels_3d: torch.Tensor,
        gt_num: int,
        img_meta: dict,
        det_token_center: torch.Tensor = None,
    ):
        '''generate prediction targets for single sample

        Args:
            cls_pred_scores (torch.Tensor): the scores(probabiliities) of classification
            bbox3d_pred (torch.Tensor): the predicted 3D bounding boxes
            gt_bboxes_3d (BaseInstance3DBoxes): GT
            gt_labels_3d (torch.Tensor): the labels
            gt_num (int): the #true_gt of current sample
            img_meta (dict): the meta info of sample
            max_gt_num (int): the max #gt of the current batch, used to pad tensors

        Returns:
            tuple[torch.Tensor]: targets
        '''
        gt_bboxes_3d = gt_bboxes_3d.to(cls_pred_logits.device)


        #calculate 3D IOU

        # valid_gt = gt_labels_3d != -1
        # gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        # gt_labels_3d = gt_labels_3d[valid_gt]
        # ipdb.set_trace()
        # generate center, dir, size target
        center_target, size_target, \
            dir_class_target, dir_res_target \
                = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d,base_xyz=det_token_center,center_res_type=self.center_res_type)
        # generate corner target
        corner_target = gt_bboxes_3d.corners

        #generate 3d bbox target
        bbox_target = gt_bboxes_3d.tensor  # TODO: bottom centered?
        bbox_target = torch.cat([gt_bboxes_3d.gravity_center, bbox_target[..., 3:]], dim=-1) # is gravity center

        # pad targets 
        # pad_num = max_gt_num - gt_labels_3d.shape[0]
        # box_label_mask = gt_labels_3d.new_zeros([max_gt_num])
        # box_label_mask[:gt_labels_3d.shape[0]] = 1

        # gt_bboxes_pad = F.pad(gt_bboxes_3d.tensor, (0, 0, 0, pad_num))
        # gt_bboxes_pad[gt_labels_3d.shape[0]:, 0:3] += 10000
        # gt_bboxes_3d = gt_bboxes_3d.new_box(gt_bboxes_pad)

        # gt_labels_3d = F.pad(gt_labels_3d, (0, pad_num))

        # center_target = F.pad(center_target, (0, 0, 0, pad_num), value=1000)
        # size_target = F.pad(size_target, (0, 0, 0, pad_num))
        # dir_class_target = F.pad(dir_class_target, (0, pad_num))
        # dir_res_target = F.pad(dir_res_target, (0, pad_num))

        # assign gt bboxes for each proposal
        if isinstance(self.matcher, _PointsInsideMatcher):
            assert det_token_center is not None
            #print(center_target.max())
            assignments = self.matcher(det_token_center, gt_bboxes_3d, gt_num, gt_labels_3d)
        else:
            assignments = self.matcher(cls_pred_logits, bbox3d_pred, gt_bboxes_3d, gt_labels_3d, gt_num, img_meta, self.bbox_coder.with_rot) 
        per_proposal_gt_inds = assignments['per_prop_gt_inds']
        proposal_matched_mask = assignments['proposal_matched_mask']

        #0. generate the iou target
        bbox_type='lidar'
        iou_calculator = BboxOverlaps3D(bbox_type)
        iou_mat = iou_calculator(bbox3d_pred, gt_bboxes_3d.tensor, mode='iou')
        iou_mat = iou_mat[0,:]
        assigned_iou_targets = torch.index_select(
            iou_mat, 0, per_proposal_gt_inds
        )  # num_query

        # 1. generate the classification target
        assigned_label_targets = torch.index_select(
            gt_labels_3d, 0, per_proposal_gt_inds
        )  # num_query 
        assigned_label_targets[proposal_matched_mask.int() == 0] = cls_pred_logits.shape[-1] - 1

        # 2. generate the center targets
        assigned_center_targets = torch.index_select(
            center_target, 0, per_proposal_gt_inds
        )

        # 3. generate the size targets
        assigned_size_targets = torch.index_select(
            size_target, 0, per_proposal_gt_inds
        )

        # 4. generate the direction targets
        assigned_dir_class_targets = torch.index_select(
            dir_class_target, 0, per_proposal_gt_inds
        )
        assigned_dir_res_targets = torch.index_select(
            dir_res_target, 0, per_proposal_gt_inds
        )

        # 5. generate the corner targets
        assigned_corner_targets = torch.index_select(
            corner_target, 0, per_proposal_gt_inds
        )

        # 6. generate 3d bbox
        assigned_3d_bbox_targets = torch.index_select(
            bbox_target, 0, per_proposal_gt_inds
        )

        return assigned_label_targets, \
                assigned_center_targets, \
                assigned_size_targets, \
                assigned_dir_class_targets, assigned_dir_res_targets, \
                assigned_corner_targets, \
                proposal_matched_mask, \
                assigned_iou_targets, \
                assigned_3d_bbox_targets

    def get_bboxes(
        self,
        bbox_preds: dict,
        input_metas: list,
        gt_labels_3d:list,
    ):
        # decode boxes
        obj_scores = bbox_preds['obj_scores']  # B x #[DET] x (num_classes + 1)
        sem_scores = obj_scores[..., :-1]
        batch_size, num_query = obj_scores.shape[:2]
        bbox3d = self.bbox_coder.decode(
            bbox_preds,  
            base_xyz = bbox_preds['det_token_center'], 
            center_res_type=self.center_res_type
        )  # B x #[DET] x 7, TODO: gravity center?

        results = list()
        for b in range(batch_size):
            # ipdb.set_trace()
            # bbox_classes = torch.argmax(obj_scores[b], dim=-1)  # #[DET]
            # bbox_fg_mask = (bbox_classes < self.num_classes)

            #using gt_labels_cls
            bbox_classes = gt_labels_3d[b]
            bbox_fg_mask_0 = (bbox_classes < self.num_classes)
            bbox_fg_mask_1 = (bbox_classes > -1)
            bbox_fg_mask = bbox_fg_mask_0==bbox_fg_mask_1

            # print(bbox_classes)

            bbox3d_fg = bbox3d[b][bbox_fg_mask]
            scores_fg = sem_scores[b][bbox_fg_mask]
            scores_fg = torch.gather(scores_fg, -1, bbox_classes[bbox_fg_mask].unsqueeze(1)).flatten()
            labels_fg = bbox_classes[bbox_fg_mask].long()
            # bbox_classes = torch.zeros(obj_scores[b].shape[0], device=obj_scores.device).long()
            # bbox3d_fg = bbox3d[b]
            # scores_fg = sem_scores[b][:, 0].flatten()
            # labels_fg = bbox_classes
            # ipdb.set_trace()
            bbox3d_fg = input_metas[b]['box_type_3d'](
                bbox3d_fg,
                box_dim=bbox3d_fg.shape[-1],
                with_yaw=self.bbox_coder.with_rot,
                origin=(0.5, 0.5, 0.5),  # that's to say, bbox3d_fg must be gravity centered
            )
            results.append((bbox3d_fg, scores_fg, labels_fg))

        return results


class _MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class _HungaryMatcher(BaseModule):
    def __init__(
        self, 
        cost_class,
        cost_objectness,
        cost_corner,
        cost_center,
        cost_iou,
        bbox_type='lidar',
        init_cfg=None
    ):
        '''Hungary Matcher used for bipartite matching

        Args:
            cost_class (_type_): _description_
            cost_objectness (_type_): _description_
            cost_corner ():
            cost_center (_type_): _description_
            init_cfg (_type_, optional): _description_. Defaults to None.
        '''
        super().__init__(init_cfg)
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_center = cost_center
        self.cost_corner = cost_corner
        self.cost_iou = cost_iou
        self.bbox_type = bbox_type.lower()

    @torch.no_grad()
    def forward(
        self, 
        cls_pred_logits: torch.Tensor,
        bbox3d_pred: torch.Tensor,
        gt_bboxes_3d: BaseInstance3DBoxes, 
        gt_labels_3d: torch.Tensor,
        gt_num: list,
        img_metas: list,
        with_rot: bool = True,
    ):
        '''Forward pass, do hungary matching
            NOTE: only used for SINGLE sample

        Args:
            cls_pred_scores (torch.Tensor): the scores(probabilities) of class branch with shape of (num_query, num_classes + 1)
            bbox3d_pred (torch.Tensor): the final prediction of heads with shape of (num_query, 7)
            gt_bboxes_3d (BaseInstance3DBoxes): ground-truth bboxes with shape of (#gt, 7)
            gt_labels_3d (torch.Tensor): ground-truth labels with shape of (#gt)
            gt_num (int): the actual gt num of each sample
            img_metas (dict): the list of image meta info
        '''
        cls_pred_scores = cls_pred_logits.softmax(-1)
        num_query = cls_pred_scores.shape[0]

        gt_labels_3d_fg = gt_labels_3d
        gt_bboxes_3d_fg = gt_bboxes_3d
        num_total_gt = gt_num

        sem_pred_prob, objectness_prob = cls_pred_scores[..., :-1], cls_pred_scores[..., -1:]

        # classification cost: #[DET] x #total_gt
        gt_box_cls_labels = gt_labels_3d_fg.unsqueeze(0).expand(num_query, num_total_gt)
        sem_mat = -1 * torch.gather(sem_pred_prob, 1, gt_box_cls_labels)  # sem_pred_prob up, cost down

        # objectness cost: #[DET] x 1
        objectness_mat = objectness_prob

        # center cost: #[DET] x #total_gt
        pred_center = bbox3d_pred[..., :3] 
        # gt_center = gt_bboxes_3d_fg.tensor[..., :3]
        # use gravity center instead 
        gt_center = gt_bboxes_3d_fg.gravity_center
        center_dist = torch.cdist(pred_center, gt_center,  1)
        center_mat = center_dist

        # corner cost:  #[DET] x #total_gt
        pred_bbox3d_warp = img_metas['box_type_3d'](
            bbox3d_pred.clone(),
            box_dim=bbox3d_pred.shape[-1],
            with_yaw=with_rot,
            origin=(0.5, 0.5, 0.5)
        )
        pred_bbox3d_corner = pred_bbox3d_warp.corners.reshape(num_query, -1)  # #[DET] x (8*3)
        # gt_bboxes_3d_corner = gt_bboxes_3d.corners.reshape(num_total_gt, -1)
        gt_bboxes_3d_corner = gt_bboxes_3d_fg.corners.reshape(num_total_gt, -1)
        corner_mat = torch.cdist(pred_bbox3d_corner, gt_bboxes_3d_corner, 1)

        # IoU cost:  # [DET] x #total_gt
        iou_calculator = BboxOverlaps3D(self.bbox_type)
        iou_mat = -1 * iou_calculator(bbox3d_pred, gt_bboxes_3d.tensor, mode='iou')

        # ipdb.set_trace()
        final_cost = (
            self.cost_class * sem_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_corner * corner_mat
            + self.cost_iou * iou_mat
        ).detach().cpu().numpy()

        # auxiliary variables useful for batched loss computation
        nprop = num_query
        per_prop_gt_inds = torch.zeros(
            [nprop], dtype=torch.int64, device=cls_pred_scores.device
        )
        proposal_matched_mask = torch.zeros(
            [nprop], dtype=torch.float32, device=cls_pred_scores.device
        )
        assign = []
        if gt_num > 0:
            assign = linear_sum_assignment(final_cost[:, : gt_num])
            assign = [
                torch.from_numpy(x).long().to(device=cls_pred_scores.device)
                for x in assign
            ]
            per_prop_gt_inds[assign[0]] = assign[1]
            proposal_matched_mask[assign[0]] = 1

        return {
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class _HungaryMatcherV2(BaseModule):
    def __init__(
        self, 
        cost_class,
        cost_bbox,
        cost_iou,
        use_focal=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        bbox_type='lidar',
        init_cfg=None
    ):
        '''Hungary Matcher used for bipartite matching

        Args:
            cost_class (_type_): _description_
            cost_objectness (_type_): _description_
            cost_corner ():
            cost_center (_type_): _description_
            init_cfg (_type_, optional): _description_. Defaults to None.
        '''
        super().__init__(init_cfg)
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_iou = cost_iou
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bbox_type = bbox_type.lower()

    @torch.no_grad()
    def forward(
        self, 
        cls_pred_logits: torch.Tensor,
        bbox3d_pred: torch.Tensor,
        gt_bboxes_3d: BaseInstance3DBoxes, 
        gt_labels_3d: torch.Tensor,
        gt_num: list,
        img_metas: list,
        with_rot: bool = True,
    ):
        '''Forward pass, do hungary matching
            NOTE: only used for SINGLE sample

        Args:
            cls_pred_scores (torch.Tensor): the scores(probabilities) of class branch with shape of (num_query, num_classes + 1)
            bbox3d_pred (torch.Tensor): the final prediction of heads with shape of (num_query, 7)
            gt_bboxes_3d (BaseInstance3DBoxes): ground-truth bboxes with shape of (#gt, 7)
            gt_labels_3d (torch.Tensor): ground-truth labels with shape of (#gt)
            gt_num (int): the actual gt num of each sample
            img_metas (dict): the list of image meta info
        '''
        num_query = cls_pred_logits.shape[0]

        if self.use_focal:
            cls_pred_scores = cls_pred_logits.sigmoid()
        else:
            cls_pred_scores = cls_pred_logits.softmax(-1)
        sem_pred_prob = cls_pred_scores[..., :-1]

        # classification cost: #[DET] x #total_gt
        if self.use_focal:
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            neg_cost_class = (1 - alpha) * (sem_pred_prob ** gamma) * (-(1 - sem_pred_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - sem_pred_prob) ** gamma) * (-(sem_pred_prob + 1e-8).log())
            sem_mat = pos_cost_class[:, gt_labels_3d] - neg_cost_class[:, gt_labels_3d] #[N,M]
        else:
            # gt_box_cls_labels = gt_labels_3d_fg.unsqueeze(0).expand(num_query, num_total_gt)
            # sem_mat = -1 * torch.gather(sem_pred_prob, 1, gt_box_cls_labels)  # sem_pred_prob up, cost down
            sem_mat = -1 * sem_pred_prob[:, gt_labels_3d]

        # bbox cost: #[DET] x #total_gt
        # tgt_bbox = torch.concat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.dims], dim=-1)
        # bbox_mat = torch.cdist(bbox3d_pred[:, :-1], tgt_bbox, p=1)
        # # orientation similarity cost:
        # pred_angle = bbox3d_pred[:, -1:]  # #[DEt] x 1
        # gt_angle = gt_bboxes_3d.tensor[:, -1:]  # #total_gt x 1
        # gt_angle = gt_angle.transpose(0, 1).contiguous().expand((pred_angle.shape[0], -1)) # #[DET] x #total_gt
        # sin_diff = torch.abs(torch.sin(pred_angle - gt_angle))
        # bbox_mat += sin_diff

        # tgt_center = gt_bboxes_3d.gravity_center
        # bbox_mat = torch.cdist(bbox3d_pred[:, :3], tgt_center, p=1)
        # tgt_size = gt_bboxes_3d.dims
        # bbox_mat = torch.cdist(bbox3d_pred[:, 3:6], tgt_size, p=1)       
        tgt_bev_center = gt_bboxes_3d.gravity_center[..., :2]
        bbox_mat = torch.cdist(bbox3d_pred[:, :2], tgt_bev_center, p=1)

        # IoU cost:  # [DET] x #total_gt
        iou_calculator = BboxOverlapsNearest3D(self.bbox_type)
        iou_mat = -1 * iou_calculator(bbox3d_pred, gt_bboxes_3d.tensor, mode='iou')

        # ipdb.set_trace()
        final_cost = (
            self.cost_class * sem_mat
            + self.cost_bbox * bbox_mat
            + self.cost_iou * iou_mat
        ).detach().cpu().numpy()

        # auxiliary variables useful for batched loss computation
        nprop = num_query
        per_prop_gt_inds = torch.zeros(
            [nprop], dtype=torch.int64, device=cls_pred_logits.device
        )
        proposal_matched_mask = torch.zeros(
            [nprop], dtype=torch.float32, device=cls_pred_logits.device
        )
        assign = []
        if gt_num > 0:
            assign = linear_sum_assignment(final_cost[:, : gt_num])
            assign = [
                torch.from_numpy(x).long().to(device=cls_pred_logits.device)
                for x in assign
            ]
            per_prop_gt_inds[assign[0]] = assign[1]
            proposal_matched_mask[assign[0]] = 1

        return {
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class _SequentialMatcher(BaseModule):
    def __init__(
        self,
        init_cfg=None,
        **kargs,
    ):
        super().__init__(init_cfg)
    
    def forward(
        self, 
        cls_pred_logits: torch.Tensor,
        bbox3d_pred: torch.Tensor,
        gt_bboxes_3d: BaseInstance3DBoxes, 
        gt_labels_3d: torch.Tensor,
        gt_nums: list,
        img_metas: list,
        *args,
    ):
        num_pred = cls_pred_logits.shape[0]
        num_gt = gt_labels_3d.shape[0]

        # assign = torch.ones((1), device=cls_pred_logits.device)
        # per_prop_gt_inds = torch.zeros((1), dtype=torch.int64, device=cls_pred_logits.device)
        # proposal_matched_mask = torch.ones((1), device=cls_pred_logits.device)
        
        per_prop_gt_inds = torch.arange(0, num_pred, dtype=torch.int64, device=cls_pred_logits.device)
        bg_pred_mask = per_prop_gt_inds >= num_gt
        per_prop_gt_inds[bg_pred_mask] = 0
        proposal_matched_mask = torch.ones((num_pred), dtype=torch.float32, device=cls_pred_logits.device)
        proposal_matched_mask[bg_pred_mask] = 0

        return {
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class _PointsInsideMatcher(BaseModule):
    def __init__(
        self,
        **kargs,
    ):
        super().__init__()

    def forward(
        self,
        points: torch.Tensor,
        gt_bboxes_3d: BaseInstance3DBoxes,
        gt_num: int = 0,
        *args,
    ):
        '''assign gt by calculate the gt_bboxes for each points

        Args:
            points (torch.Tensor): M x 3 
            gt_bboxes (torch.Tensor): N x 7 (bottom centered, decoded bbox)
        '''

        num_query = points.shape[0]

        per_prop_gt_inds = points.new_zeros((num_query, ), dtype=torch.int64)
        proposal_matched_mask = points.new_zeros((num_query), dtype=torch.float32)

        if gt_num > 0:
            point_mask = gt_bboxes_3d.points_in_boxes_all(points)  # M x N
            gt_inds = point_mask.argmax(-1)  # M, 
            proposal_matched_mask = (point_mask.sum(-1) > 0)  # M,
            per_prop_gt_inds = gt_inds
            #print(per_prop_gt_inds)
        return {
            'per_prop_gt_inds': per_prop_gt_inds,
            'proposal_matched_mask': proposal_matched_mask
        }