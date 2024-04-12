_base_ = [
    './votenet_16x8_sunrgbd-3d-10class.py'
]
# dataset settings
load_interval = 5  # 20%

data = dict(
    train=dict(
        times=5 * load_interval,
        dataset=dict(
            load_interval=load_interval,
        )
    ),
)