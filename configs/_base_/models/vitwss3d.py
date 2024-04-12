model = dict(
    type='ViTWSS3D',
    backbone = dict(
        type='PointTransformer',
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        group_k=32,
        patch_num=2048,
        encoder_dims=256,
        det_token_num=100,
    ),
    bbox_head = dict(
        type='ViTWSS3DHead',
        # num_classes
        in_channels=384,
        # bbox_coder
        num_mlp_layers=3,
        mlp_channel=384,
        # matcher_cfg
    )
)