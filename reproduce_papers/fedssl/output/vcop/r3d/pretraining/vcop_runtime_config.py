dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='VCOPDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/train_split_1.json'),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg'),
        test_mode=False,
        transform_cfg=[
            dict(type='GroupScale', scales=[(171, 128)]),
            dict(type='GroupRandomCrop', out_size=112),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ],
        clip_len=16,
        min_interval=8,
        max_interval=8,
        tuple_len=3,
        data_dir='/home/root/yasar/Dataset/'),
    val=dict(
        type='VCOPDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/test_split_1.json'),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg'),
        test_mode=True,
        transform_cfg=[
            dict(type='GroupScale', scales=[(171, 128)]),
            dict(type='GroupCenterCrop', out_size=112),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ],
        clip_len=16,
        min_interval=8,
        max_interval=8,
        tuple_len=3,
        data_dir='/home/root/yasar/Dataset/'))
total_epochs = 1
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[100, 200])
checkpoint_config = dict(interval=1, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './output/vcop/r3d/pretraining/'
model = dict(
    type='VCOP',
    backbone=dict(
        type='R3D',
        depth=18,
        num_stages=4,
        stem=dict(
            temporal_kernel_size=3,
            temporal_stride=1,
            in_channels=3,
            with_pool=False),
        down_sampling=[False, True, True, True],
        channel_multiplier=1.0,
        bottleneck_multiplier=1.0,
        with_bn=True,
        pretrained=None),
    vcop_head=dict(
        in_channels=512, tuple_len=3, hidden_channels=512, dropout_ratio=0.5))
gpus = 1
