_base_ = [
    './votenet_16x8_sunrgbd-3d-10class.py'
]

# dataset settings
data_root = 'data/sunrgbd_80pseudo/'
data = dict(
    train=dict(
        dataset=dict(
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            )),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
    )
)