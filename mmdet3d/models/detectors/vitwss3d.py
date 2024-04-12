import ipdb
import os
import torch
from torch.nn.utils import clip_grad

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.core.visualizer.show_result import show_result

from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector


@DETECTORS.register_module()
class ViTWSS3D(SingleStage3DDetector):
    def __init__(
        self, 
        backbone, 
        neck=None, 
        bbox_head=None, 
        train_cfg=None, 
        test_cfg=None, 
        init_cfg=None, 
        pretrained=None
    ):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg, pretrained)

    def extract_feat(
        self, 
        points, 
        img_metas=None, 
        gt_bboxes_3d=None, 
        gt_labels_3d = None,
        img=None
    ):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        x = self.backbone(points, gt_bboxes_3d, gt_labels_3d, img=img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, points, img_metas, gt_bboxes_3d=None):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img_meta, gt_bbox)
            for pts, img_meta, gt_bbox in zip(points, img_metas, gt_bboxes_3d)
        ]

    def forward_train(
        self,
        points: list,
        img_metas: list,
        gt_bboxes_3d: list,
        gt_labels_3d: list,
        img: torch.Tensor = None,
    ):
        '''forward pass for training

        Args:
            points (list): the points of batch
            img_metas (list): the image meta info of batch
            gt_bboxes_3d (list): the gt bboxes of batch
            gt_labels_3d (list): the gt labels of batch
            img (torch.Tensor): the images of shape (N, C, H, W)
        '''
        # TODO: [warning] using gt info here
        points_tensor = torch.stack(points)
        
        # ipdb.set_trace()

        backbone_feats = self.extract_feat(
            points_tensor, 
            gt_bboxes_3d=gt_bboxes_3d, 
            gt_labels_3d=gt_labels_3d,
            img=img
        )
        bbox_preds = self.bbox_head(backbone_feats)
        losses = self.bbox_head.loss(
            bbox_preds,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas
        )

        # detect the NaN
        for loss in losses.values():
            if torch.isnan(loss):
                error_msg = 'Found NaN!, the frame ids are:\n'
                for meta_info in img_metas:
                    error_msg += f'pts_filename: {meta_info["pts_filename"]}\n'
                print(error_msg)
                ipdb.set_trace()
                for idx in range(len(points)):
                    pts_filename = img_metas[idx]['pts_filename']
                    pts_filename = os.path.basename(pts_filename)
                    pts_filename = os.path.splitext(pts_filename)[0]
                    # ipdb.set_trace()
                    show_result(
                        points[idx], 
                        gt_bboxes=gt_bboxes_3d[idx].tensor.cpu().numpy(), 
                        pred_bboxes=None,
                        out_dir='/home/dyzhang/YOLOS3d/error_vis_results',
                        filename=pts_filename
                    )
                raise ValueError

        # visualize the input points
        # bbox_list = self.bbox_head.get_bboxes(bbox_preds, img_metas)

        # # ipdb.set_trace()
        
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in bbox_list
        # ]
        # for idx in range(len(points)):
        #     pts_filename = img_metas[idx]['pts_filename']
        #     pts_filename = os.path.basename(pts_filename)
        #     pts_filename = os.path.splitext(pts_filename)[0]
        #     # ipdb.set_trace()
        #     show_result(
        #         points[idx], 
        #         gt_bboxes=gt_bboxes_3d[idx].tensor.cpu().numpy(), 
        #         pred_bboxes=bbox_results[idx]['boxes_3d'].tensor.cpu().detach().numpy(),
        #         out_dir='/home/dyzhang/YOLOS3d/vis_results',
        #         filename=pts_filename
        #     )

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d = None):
        '''forward pass for testing 

        Args:
            points (list): the points of batch
            img_metas (list): the image meta info of batch
            rescale (bool, optional): _description_. Defaults to False.
        '''
        points_tensor = torch.stack(points)
        gt_bboxes_3d = gt_bboxes_3d[0] # only one batch
        gt_labels_3d = gt_labels_3d[0] # only one batch

        #remove the do not care boxes
        gt_bboxes_3d[0] = gt_bboxes_3d[0][gt_labels_3d[0] >= 0]
        gt_labels_3d[0] = gt_labels_3d[0][gt_labels_3d[0] >= 0]

        backbone_feats = self.extract_feat(
            points_tensor, 
            gt_bboxes_3d=gt_bboxes_3d, 
            gt_labels_3d=gt_labels_3d,
            img=imgs,
        )
        bbox_preds = self.bbox_head(backbone_feats)

        gt_num = gt_bboxes_3d[0].center.shape[0]
        #gt_num = gt_bboxes_3d[0][gt_labels_3d[0] >= 0].center.shape[0] # remove the do not care boxes


        for key in bbox_preds.keys():
            bbox_preds[key] = bbox_preds[key][:,0:gt_num,:] # choose the positive

        bbox_list = self.bbox_head.get_bboxes(bbox_preds, img_metas, gt_labels_3d)

        # ipdb.set_trace()
        
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        # for bboxes, scores, labels in bbox_list:
        #     print(scores, labels)

        #set the score as 1
        # bbox_results = [
        #     bbox3d2result(bboxes, torch.full([scores.shape[0]], 1).cuda(), labels)
        #     for bboxes, scores, labels in bbox_list
        # ]

        # file_name = img_metas[0]['pts_filename'].split('/')[-1]
        # bbox_results[0]['file_name'] = file_name
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_tensor = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_tensor, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            bbox_preds = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                bbox_preds, img_meta)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
    