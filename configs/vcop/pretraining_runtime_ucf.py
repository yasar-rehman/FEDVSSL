dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True
# Define a dictionary that contains training data, validation data and test data descriptions
data = dict(
    videos_per_gpu=4, # actual is 4, set to 8 because only 4 gpu's
    workers_per_gpu=4, # actual is 4, set to 8 to speedup
    train=dict(
        type='VCOPDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='Kinetics-processed/zips/{}.zip',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='Kinetics-processed/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
        test_mode=False,
        transform_cfg=[
                dict(type='GroupScale', scales=[(171, 128)]),
                dict(type='GroupRandomCrop', out_size=112),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
        ],
        clip_len=16,
        min_interval=8,
        max_interval=8,
        tuple_len=3,
    ),
    val=dict(
        type='VCOPDataset',
        name='ucf101_test_split1',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/test_split_1.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
        test_mode=True,
        transform_cfg=[
            dict(type='GroupScale', scales=[(171, 128)]),
            dict(type='GroupCenterCrop', out_size=112),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ],
        clip_len=16,
        min_interval=8,
        max_interval=8,
        tuple_len=3,
    )
)


# optimizer
total_epochs = 1
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 60],
)
checkpoint_config = dict(interval=1, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)
