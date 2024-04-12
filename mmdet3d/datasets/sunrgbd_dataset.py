# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from os import path as osp

import ipdb
import numpy as np
import tempfile

import mmcv
from mmdet3d.core import show_multi_modality_result, show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.core import eval_map
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class SUNRGBDDataset(Custom3DDataset):
    r"""SUNRGBD Dataset.

    This class serves as the API for experiments on the SUNRGBD Dataset.

    See the `download page <http://rgbd.cs.princeton.edu/challenge.html>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
            'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str, optional): Filename of point clouds.
                - file_name (str, optional): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - calib (dict, optional): Camera calibration info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        assert info['point_cloud']['lidar_idx'] == info['image']['image_idx']
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_lidar']:
            pts_filename = osp.join(self.data_root, info['pts_path'])
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_filename = osp.join(
                osp.join(self.data_root, 'sunrgbd_trainval'),
                info['image']['image_path'])
            input_dict['img_prefix'] = None
            input_dict['img_info'] = dict(filename=img_filename)
            calib = info['calib']
            rt_mat = calib['Rt']
            # follow Coord3DMode.convert_point
            rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                               ]) @ rt_mat.transpose(1, 0)
            depth2img = calib['K'] @ rt_mat
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            # if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
            #     return None
        return input_dict

    def prepare_test_data(self, index):
        """Prepare data for testing

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        original_test_model = self.test_mode
        self.test_mode = False
        input_dict = self.get_data_info(index)
        self.test_mode = original_test_model
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        if self.modality['use_camera']:
            if info['annos']['gt_num'] != 0:
                gt_bboxes_2d = info['annos']['bbox'].astype(np.float32)
            else:
                gt_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
            anns_results['bboxes'] = gt_bboxes_2d
            anns_results['labels'] = gt_labels_3d

        return anns_results

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            # scale colors to [0, 255]
            points = points.numpy()
            points[:, 3:] *= 255

            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_result(points, gt_bboxes.copy(), pred_bboxes.copy(), out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera']:
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                pred_bboxes = DepthInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                gt_bboxes = DepthInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    gt_bboxes,
                    pred_bboxes,
                    None,
                    out_dir,
                    file_name,
                    box_mode='depth',
                    img_metas=img_metas,
                    show=show)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 iou_thr_2d=(0.5, ),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            iou_thr (list[float], optional): AP IoU thresholds for 3D
                evaluation. Default: (0.25, 0.5).
            iou_thr_2d (list[float], optional): AP IoU thresholds for 2D
                evaluation. Default: (0.5, ).
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        # evaluate 3D detection performance
        if isinstance(results[0], dict):
            return super().evaluate(results, metric, iou_thr, logger, show,
                                    out_dir, pipeline)
        # evaluate 2D detection performance
        else:
            eval_results = OrderedDict()
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            iou_thr_2d = (iou_thr_2d) if isinstance(iou_thr_2d,
                                                    float) else iou_thr_2d
            for iou_thr_2d_single in iou_thr_2d:
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr_2d_single,
                    dataset=self.CLASSES,
                    logger=logger)
                eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
            return eval_results

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
                boxes_3d:
                scores_3d:
                labels_3d:
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving json
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = pklfile_prefix
            pklfile_prefix = osp.join(pklfile_prefix, 'results')
        out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)

        if submission_prefix is not None:  # save results into .txt files
            self.bbox2result_sunrgbd(outputs, self.CLASSES, submission_prefix)
        return outputs, tmp_dir

    def bbox2result_sunrgbd(self,
                          net_outputs,
                          class_names,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[dict]): List of dict storing the
                inferenced bounding boxes and scores.
                boxes_3d:
                scores_3d:
                labels_3d:
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        print(f'\nConverting prediction to SUN-RGBD format, saving to {submission_prefix}')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            anno = {
                'name': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            # ipdb.set_trace()
            if len(pred_dicts['boxes_3d']) > 0:
                box_2d_preds = self.data_infos[idx]['annos']['bbox']  # do not change the 2d boxes
                box_preds = pred_dicts['boxes_3d'].tensor.numpy()
                box_preds[:, 2] += box_preds[:, 5] * 0.5
                scores = pred_dicts['scores_3d'].numpy()
                label_preds = pred_dicts['labels_3d'].numpy()

                for box, bbox, score, label in zip(
                        box_preds, box_2d_preds, scores,
                        label_preds):
                    anno['name'].append(class_names[int(label)])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
            with open(curr_file, 'w') as f:
                bbox = anno['bbox']
                loc = anno['location']
                dims = anno['dimensions'] * 0.5  # lhw -> hwl
                rotation_y = anno['rotation_y']
                # x_coord = np.ones_like(rotation_y)
                # y_coord = np.tan(rotation_y)  # TODO: TypeError: only size-1 arrays can be converted to Python scalars

                for idx in range(len(bbox)):
                    rot = rotation_y[idx]
                    x_coord = 1 if rot > -np.math.pi / 2 and rot < np.math.pi / 2 else -1
                    y_coord = x_coord * np.math.tan(rot)
                    print(
                        '{} '  # class name
                        '{:.6f} {:.6f} {:.6f} {:.6f} '  # 2d box
                        '{:.6f} {:.6f} {:.6f} '  # location
                        '{:.6f} {:.6f} {:.6f} '  # dims
                        '{:.6f} {:.6f}'.format(  # (x-coord, y-coord) s.t tan(rotation_y) = y-coord / x-coord
                            anno['name'][idx], 
                            bbox[idx][0], bbox[idx][1], bbox[idx][2] - bbox[idx][0], bbox[idx][3] - bbox[idx][1], 
                            loc[idx][0], loc[idx][1], loc[idx][2], 
                            dims[idx][1], dims[idx][0], dims[idx][2], 
                            x_coord, y_coord
                        ),
                        file=f)
