_base_ = '../pretraining_runtime_kinetics_fed.py'

# work_dir = './output/speednet/r3d_18_kinetics/pretraining/'

model = dict(
    type='SpeedNet',
    backbone=dict(
        type='R3D',
        depth=18,
        num_stages=4,
        stem=dict(
            temporal_kernel_size=3,
            temporal_stride=1,
            in_channels=3,
            with_pool=False,
        ),
        down_sampling=[False, True, True, True],
        channel_multiplier=1.0,
        bottleneck_multiplier=1.0,
        with_bn=True,
        pretrained=None,
    ),
    st_module=dict(
        spatial_type='avg',
        temporal_size=2,  # 16//8
        spatial_size=7),
    cls_head=dict(
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=512,
        init_std=0.001,
        num_classes=7
    )
)

