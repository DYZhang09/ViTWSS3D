_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/models/vitwss3d.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/cyclic_40e.py'
]
find_unused_parameters=True
# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
#input_modality = dict(use_lidar=True, use_camera=True)

# multi-class
class_names = ['Car', 'Pedestrian', 'Cyclist']
db_filter_cfg = dict(Car=5, Pedestrian=5, Cyclist=5)
# db_sample_cfg = dict(Car=20, Pedestrian=15, Cyclist=15)
db_sample_cfg = dict(Car=15, Pedestrian=8, Cyclist=8)
mean_sizes=[
    [3.9, 1.6, 1.56], 
    [0.8, 0.6, 1.73], 
    [1.76, 0.6, 1.73]
]

# single class
# class_names = ['Car']
# db_filter_cfg = dict(Car=5)
# db_sample_cfg = dict(Car=20)
# mean_sizes= [
#     [3.9, 1.6, 1.56], 
# ]


db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train_1_10.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=db_filter_cfg,
    ),
    sample_groups=db_sample_cfg,
    classes=class_names)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox = True, with_label = True),
    dict(
        type='Resize',
        img_scale=(1333, 384),
        keep_ratio=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectSample', db_sampler=db_sampler, sample_2d=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='PointSample', num_points=16384, sample_range=40.0),
    dict(type='PointSample', num_points=16384),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',  'img'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=False,
        with_seg_3d=False),
    dict(
        type='MultiScaleFlipAug3D',
        # img_scale=(1333, 800),
        img_scale=(1333, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=16384, sample_range=40.0),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=40,
        dataset=dict(pipeline=train_pipeline, classes=class_names,
                     ann_file=data_root + 'kitti_infos_train_1_10.pkl',
                     use_pseudo_label = False
                     )),
    val=dict(pipeline=test_pipeline, classes=class_names,ann_file=data_root + 'kitti_infos_train_9_10.pkl'),
    test=dict(pipeline=test_pipeline, classes=class_names, ann_file=data_root + 'kitti_infos_train_9_10.pkl'))

general_dim = 768
# model settings
model = dict(
    type='ViTWSS3D',
    backbone=dict(
        in_channels=4,
        trans_dim=general_dim,
        depth=12,
        num_heads=6,
        patch_num=2048,
        det_token_num=100,
        bbox_mean_size=mean_sizes,
        det_token_init_type='gt',
        det_init_k=32,  # the k of knn is 32
        use_random_gt = True,
        random_gt_shift = [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1], #random shift for [x1,x2]...[z1,z2]
        dynamic_det_token_num = False,
        pretrained = True, # using pre-trained for MAE_vit
        share_mlp  = True,
        encoding_gt_label = False,
        encoding_size_prior = False,
        # use image tokens
        use_img_tokens = False,
        img_channels = 3,
        img_size = (1280, 384),
        img_patch_size = 32,
        pretrained_model = './mae_finetuned_vit_base.pth', # mae_finetuned_vit_base(768 dim,12 depth,12 heads), mae_finetuned_vit_large(1024,24,16), mae_finetuned_vit_huge(1280,32,16),
        type='PointTransformer_MAE', #PointTransformer for dyzhang, PointTransformer_MAE for dkliang
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
            point_cloud_range=point_cloud_range,
            with_rot=True,
        ),
        matcher_cfg=dict(
            type='PointsInsideMatcher',
            cost_class=2.0,
            cost_bbox=5.0,
            cost_iou=2.0,
            use_focal=True,
            bbox_type='LiDAR'
        ),
        sem_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='none',
            gamma=2.0,
            alpha=0.25,
            activated=False,
            loss_weight=2.0
        ),
        center_res_type='offset',  # 'direct' or 'offset', directly regressing the coordinates or regressing offset
        center_loss=dict(
            type='SmoothL1Loss',
            reduction='none',
            loss_weight=1.0,
        ),
        size_loss=dict(
            type='SmoothL1Loss',
            reduction='none',
            loss_weight=1.0
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
            loss_weight=0.0
        ),
    ),
    train_cfg=None,
    test_cfg=None,
)


# optimizer settings
lr = 1e-4
# # cosine annealing
# lr_config = dict(
#     warmup_iters=100,
# )
optimizer = dict(lr=lr, betas=(0.95, 0.85))
optimizer_config = dict()
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(max_keep_ckpts=10)
# load_from = 'pretrain_weights/point-bert_converted.pth'
evaluation = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])