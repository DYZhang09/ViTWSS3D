_base_ = [
    './vitwss3d_small_sunrgbd.py'
]
# dataset settings
load_interval = 5  # 20%

data = dict(
    train=dict(
        times=2 * load_interval,
        dataset=dict(
            load_interval=load_interval,
        )
    ),
)

