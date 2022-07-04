_base_ = './pretraining_runtime_ucf.py'

data = dict(
    train=dict(
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='Kinetics-processed/annotations/train_in_official.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='Kinetics-processed/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
    )
)

# optimizer
total_epochs = 90
