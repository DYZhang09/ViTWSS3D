_base_ = [
    './fcaf3d_8x2_sunrgbd-3d-10class.py'
]
# dataset settings
data_root = 'data/sunrgbd_80pseudo_sizeL5/'
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