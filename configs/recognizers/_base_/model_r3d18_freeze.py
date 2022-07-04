model = dict(
    type='TSN',
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
        zero_init_residual=False,
        frozen=5,
        pretrained=None,
    ),
    st_module=dict(
        spatial_type='avg',
        temporal_size=-1,  # 16//8 # for linear probe this should be set to -1 for fine-tuning set it to 2
        spatial_size=-1), # for linear probe this should be set to -1, for fine-tuning set it to 7
    cls_head=dict(
        with_avg_pool=False, #
        use_final_bn=True, # true only for linear probe
        use_l2_norm=True, # true only for linear probe
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0, # zero only for linear probe. for the fine-tuning it should be set to 0.8
        in_channels=512,
        init_std=0.001,
        num_classes=101
    )
)
