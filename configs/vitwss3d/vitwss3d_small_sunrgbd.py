_base_ = [
    '../_base_/schedules/schedule_3x.py',
    '../_base_/models/vitwss3d.py',
    '../_base_/default_runtime.py',
]
find_unused_parameters=True
# mean_sizes=[
#     [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
#     [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
#     [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
#     [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
#     [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
# ]
mean_sizes = None

# dataset settings
dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/sunrgbd/':
#         's3://openmmlab/datasets/detection3d/sunrgbd_processed/',
#         'data/sunrgbd/':
#         's3://openmmlab/datasets/detection3d/sunrgbd_processed/'
#     }))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', file_client_args=file_client_args),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=False),
    dict(type='PointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='PointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='Depth',
            file_client_args=file_client_args)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args))



general_dim = 384
# model settings
model = dict(
    type='ViTWSS3D',
    backbone=dict(
        type='PointTransformer_MAE', #PointTransformer for dyzhang, PointTransformer_MAE for dkliang
        in_channels=3,
        trans_dim=general_dim,
        depth=12,
        num_heads=6,
        patch_num=2048,
        det_token_num=300,
        group_k=32,
        bbox_mean_size=mean_sizes,
        det_token_init_type='gt',
        # det_init_k=32,  # the k of knn is 32
        use_random_gt = False,
        random_gt_shift = [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1], #random shift for [x1,x2]...[z1,z2]
        dynamic_det_token_num = False,
        pretrained = True, # using pre-trained for MAE_vit
        share_mlp  = True,
        encoding_gt_label = False,
        encoding_size_prior = False,
        # use image tokens
        use_img_tokens = False,
        # img_channels = 3,
        # img_size = (1280, 384),
        # img_patch_size = 32,
        pretrained_model = './mae_finetuned_vit_small.pth', # mae_finetuned_vit_base(768 dim,12 depth,12 heads), mae_finetuned_vit_large(1024,24,16), mae_finetuned_vit_huge(1280,32,16),
    ),
    bbox_head=dict(
        type='ViTWSS3DHead',
        in_channels=general_dim,
        num_mlp_layers=3,
        mlp_channel=general_dim,
        num_classes=len(class_names),
        bbox_coder=dict(
            type='ViTWSS3DBBoxCoder',
            num_dir_bins=12,
            # point_cloud_range=point_cloud_range,
            with_rot=True,
        ),
        matcher_cfg=dict(
            # type='PointsInsideMatcher',
            type='SequentialMatcher',
            # cost_class=2.0,
            # cost_bbox=5.0,
            # cost_iou=2.0,
            # use_focal=True,
            # bbox_type='LiDAR'
        ),
        sem_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='none',
            gamma=2.0,
            alpha=0.25,
            activated=False,
            loss_weight=1.0
        ),
        center_res_type='direct',  # 'direct' or 'offset', directly regressing the coordinates or regressing offset
        center_loss=dict(
            type='L1Loss',
            reduction='none',
            loss_weight=1.0,
        ),
        size_loss=dict(
            type='L1Loss',
            reduction='none',
            loss_weight=5.0
        ),
        dir_cls_loss=dict(
            type='CrossEntropyLoss',
            reduction='none',
            loss_weight=1.0
        ),
        dir_res_loss=dict(
            type='SmoothL1Loss',
            reduction='none',
            loss_weight=1.0
        ),
        corner_loss=dict(
            type='SmoothL1Loss',
            reduction='none',
            loss_weight=0,
        ),
        iou_loss=dict(
            type='RotatedIoU3DLoss',
            reduction='none',
            loss_weight=1.0
        ),
    ),
    train_cfg=None,
    test_cfg=None,
)

evaluation = dict(interval=2)

lr = 1e-4
optimizer = dict(
    lr=lr,
    weight_decay=0.0005,
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# lr_config = dict(policy='step', warmup=None, step=[56, 112, 148])
# runner = dict(type='EpochBasedRunner', max_epochs=160)
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])