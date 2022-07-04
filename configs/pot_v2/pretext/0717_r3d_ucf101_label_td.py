_base_ = ['./_base_/model_r3d_18.py',
          './_base_/runtime_ucf101.py']
work_dir = './output/pot_v2/pretext/0717_r3d_ucf101_label_td'

model = dict(
    pot_head=dict(with_label_head=True)
)

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='POTPatchDatasetV2',
        name='ucf101_train_split1',
        backend=dict(type='ZipBackend'),
        frame_sampler=dict(type='RandomFrameSampler', clip_len=16, strides=[1, 2, 3, 4], temporal_jitter=True),
        transform_cfg=dict(
            type='Compose',
            transform_cfgs=[
                dict(type='GroupScale', scales=[(149, 112), (171, 128), (192, 144)]),
                dict(type='GroupRandomCrop', out_size=112),
                dict(
                    type='PatchMaskV3',
                    region_sampler=dict(
                        type='TemporalDiffRegionSampler',
                        patch_scales=[16, 24, 28, 32, 48, 64],
                        patch_ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                        scale_jitter=0.18,
                        num_patches=3,
                    ),
                    key_frame_probs=[0.5, 0.3, 0.2],
                    loc_velocity=3,
                    size_velocity=0.025,
                    label_prob=0.8
                ),
                dict(type='RandomHueSaturation', prob=0.25, hue_delta=12, saturation_delta=0.1),
                dict(type='DynamicBrightness', prob=0.5, delta=30, num_key_frame_probs=(0.7, 0.3)),
                dict(type='DynamicContrast', prob=0.5, delta=0.12, num_key_frame_probs=(0.7, 0.3)),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
        ),
    ),
)