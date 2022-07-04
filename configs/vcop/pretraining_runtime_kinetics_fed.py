_base_ = './pretraining_runtime_ucf.py'

data = dict(
    train=dict(
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='Kinetics-processed/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
    )
)

# optimizer
total_epochs = 1
optimizer = dict(type='SGD', lr=0.001, momentum=0, weight_decay=0.0001)  # same config as official
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 60]
)
checkpoint_config = dict(interval=1, max_keep_ckpts=-1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
