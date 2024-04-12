import ipdb
import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from mmdet3d.core.bbox.structures.base_box3d import BaseInstance3DBoxes
from .anchor_free_bbox_coder import AnchorFreeBBoxCoder


@BBOX_CODERS.register_module()
class ViTWSS3DBBoxCoder(AnchorFreeBBoxCoder):
    def __init__(self, num_dir_bins, point_cloud_range=None, with_rot=True, mean_sizes=None):
        super().__init__(num_dir_bins, with_rot)
        self.point_cloud_range = point_cloud_range  # [x_min, y_min, z_min, x_max, y_max, z_max]
        if mean_sizes is not None:
            self.mean_sizes = torch.from_numpy(np.array(mean_sizes, dtype=np.float32))
        else:
            self.mean_sizes = None

    def _calculate_range_center_dimensions(self, device):
        centers = list()
        dimensions = list()
        for i in range(3):  # x, y, z
            centers.append(sum(self.point_cloud_range[i::3]) / 2)
            dimensions.append((self.point_cloud_range[i + 3] - self.point_cloud_range[i]) / 2)
        range_center = torch.tensor(centers, device=device)  # 3, 
        range_dimensions = torch.tensor(dimensions, device=device)  # 3.
        return range_center, range_dimensions
    
    def encode(self, gt_bboxes_3d: BaseInstance3DBoxes, gt_labels_3d, base_xyz = None, center_res_type='direct'):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes
                with shape (n, 7).
            gt_labels_3d (torch.Tensor): Ground truth classes.

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center  # n, 3
        if center_res_type=='direct':
            if self.point_cloud_range is not None:
                range_center, range_dimensions = self._calculate_range_center_dimensions(gt_labels_3d.device)
                center_target -= range_center.unsqueeze(0)
                center_target /= range_dimensions.unsqueeze(0)  # normalized
        elif center_res_type=='offset':
            range_center, range_dimensions = self._calculate_range_center_dimensions(gt_labels_3d.device)
            center_target = base_xyz.detach().clone()
            center_target -= range_center.unsqueeze(0)
            center_target /= range_dimensions.unsqueeze(0)  # normalized

            original_center_target = gt_bboxes_3d.gravity_center.clone()
            original_center_target -= range_center.unsqueeze(0)
            original_center_target /= range_dimensions.unsqueeze(0)  # normalized

            center_target[0:gt_bboxes_3d.gravity_center.shape[0], :] =  original_center_target - center_target[0:gt_bboxes_3d.gravity_center.shape[0], :]
        elif center_res_type == 'distance_offset':
            range_center, range_dimensions = self._calculate_range_center_dimensions(gt_labels_3d.device)
            center_target -= range_center.unsqueeze(0)
            center_target /= range_dimensions.unsqueeze(0)  # normalized

            size_res_target = gt_bboxes_3d.dims
            size_res_target /=range_dimensions.unsqueeze(0)

            l = base_xyz[:, 0] - center_target[:, 0] + 0.5 * size_res_target[:, 0]
            t = base_xyz[:, 1] - center_target[:, 1] + 0.5 * size_res_target[:, 1]
            h0 = base_xyz[:, 2] - center_target[:, 2] + 0.5 * size_res_target[:, 2]
            r = center_target[:, 0] + 0.5 * size_res_target[:, 0] - base_xyz[:, 0]
            b = center_target[:, 1] + 0.5 * size_res_target[:, 1] - base_xyz[:, 1]
            h1 = center_target[:, 2] + 0.5 * size_res_target[:, 2] - base_xyz[:, 2]

            reg_target = torch.stack([l, t, h0, r, b, h1], dim=1)

        else:
            raise NotImplementedError(f'not implemented center_res_type {center_res_type}')

        # generate bbox size target
        if self.mean_sizes is None:  # Not using mean_sizes
            size_res_target = gt_bboxes_3d.dims / 2
        else:
            self.mean_sizes = self.mean_sizes.to(gt_labels_3d.device)
            target_mean_size = self.mean_sizes[gt_labels_3d]
            
            # ipdb.set_trace()
            size_res_target = gt_bboxes_3d.dims - target_mean_size

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            (dir_class_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d.yaw)
            dir_res_target /= (2 * np.pi / self.num_dir_bins)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        return (center_target, size_res_target, dir_class_target,
                dir_res_target)
    
    def decode(self, bbox_out, base_xyz = None, center_res_type='direct'):
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - center: predicted gravity center of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size: predicted bbox size.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (batch, n, 7).
        """
        center_pred = bbox_out['center']
        if center_res_type == 'direct':
            if self.point_cloud_range is not None:
                range_center, range_dimensions = self._calculate_range_center_dimensions(center_pred.device)
                center = center_pred * range_dimensions.reshape(1, 1, -1) + range_center.reshape(1, 1, -1)  # unnormed
            else:
                center = center_pred
        elif center_res_type == 'offset': #using offset
            range_center, range_dimensions = self._calculate_range_center_dimensions(center_pred.device)
            base_xyz = base_xyz.detach().clone()
            base_xyz -= range_center.unsqueeze(0)
            base_xyz /= range_dimensions.unsqueeze(0)  # normalized

            center_pred = base_xyz + center_pred
            center = center_pred * range_dimensions.reshape(1, 1, -1) + range_center.reshape(1, 1, -1)  # unnormed
        else:
            raise NotImplementedError(f'not implemented center_res_type {center_res_type}')

        batch_size, num_proposal = center.shape[:2]

        # decode heading angle
        if self.with_rot:
            dir_class = torch.argmax(bbox_out['dir_class'], -1)
            dir_res = torch.gather(bbox_out['dir_res'], 2,
                                   dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(
                batch_size, num_proposal, 1)
        else:
            dir_angle = center.new_zeros(batch_size, num_proposal, 1)

        # decode bbox size
        bbox_size_encoded = bbox_out['size']
        if self.mean_sizes is None:
            bbox_size = torch.clamp(bbox_size_encoded * 2, min=0.1)
        else:
            self.mean_sizes = self.mean_sizes.to(center_pred.device)
            bbox_scores = bbox_out['obj_scores']
            bbox_cls = torch.argmax(bbox_scores, -1).long()  # B x N
            # ipdb.set_trace()
            bbox_cls[bbox_cls == bbox_scores.shape[-1] - 1] = 0  # set bg bbox to 0 to prevent error
            mean_size = self.mean_sizes[bbox_cls.flatten()].reshape(batch_size, num_proposal, -1)  # B x N x 3
            bbox_size = bbox_size_encoded + mean_size
            bbox_size = torch.clamp(bbox_size, min=0.1)

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d


    def split_pred(self, cls_preds, reg_preds, base_xyz, center_res_type='direct'):
        """Split predicted features to specific parts.

        Args:
            cls_preds (torch.Tensor): Class predicted features to split.
            reg_preds (torch.Tensor): Regression predicted features to split.
            base_xyz (torch.Tensor): Coordinates of points.

        Returns:
            dict[str, torch.Tensor]: Split results.
        """
        results = {}
        results['obj_scores'] = cls_preds

        start, end = 0, 0
        reg_preds_trans = reg_preds.transpose(2, 1)

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        # results['center_offset'] = reg_preds_trans[..., start:end]
        if center_res_type == 'direct':
            results['center'] = reg_preds_trans[..., start:end]
        elif center_res_type == 'offset':
            results['center'] = reg_preds_trans[..., start:end]
            #results['center'] = base_xyz.detach() + reg_preds_trans[..., start:end]
        elif center_res_type == 'distance_offset':
            results['center'] = reg_preds_trans[..., start:end]
        else:
            print("\n*****Unknown center_res_type****\n")
            exit()
        start = end

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['size'] = reg_preds_trans[..., start:end]
        start = end

        # decode direction
        end += self.num_dir_bins
        results['dir_class'] = reg_preds_trans[..., start:end]
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end]
        start = end

        results['dir_res_norm'] = dir_res_norm
        results['dir_res'] = dir_res_norm * (2 * np.pi / self.num_dir_bins)

        return results
