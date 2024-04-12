_base_ = [
    './fcaf3d_8x2_sunrgbd-3d-10class.py'
]
# dataset settings
load_interval = 5  # 20%

data = dict(
    train=dict(
        times=3 * load_interval,
        dataset=dict(
            load_interval=load_interval,
        )
    ),
)