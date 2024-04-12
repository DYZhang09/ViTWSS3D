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

# multi-class
class_names = ['Car', 'Pedestrian', 'Cyclist']
db_filter_cfg = dict(Car=5, Pedestrian=5, Cyclist=5)
db_sample_cfg = dict(Car=20, Pedestrian=15, Cyclist=15)
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
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=db_filter_cfg,
    ),
    sample_groups=db_sample_cfg,
    classes=class_names)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectSample', db_sampler=db_sampler),
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=False,
        with_seg_3d=False),
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
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=16384, sample_range=40.0),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

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
        use_random_gt = False,
        dynamic_det_token_num = False,
        pretrained=True, # using pre-trained for MAE_vit
        pretrained_model = './mae_finetuned_vit_base.pth', # mae_finetuned_vit_base(768 dim,12 depth,12 heads), mae_finetuned_vit_large(1024,24,16), mae_finetuned_vit_huge(1280,32,16)
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
            type='HungaryMatcherV2',
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
        center_loss=dict(
            type='L1Loss',
            reduction='none',
            loss_weight=1.0,
        ),
        size_loss=dict(
            type='L1Loss',
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
# load_from = 'pretrain_weights/point-bert_converted.pth'
evaluation = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])