_base_ = './0723_r3d_ucf101_wo_label_circle.py'
work_dir = './output/pot_v2/pretext/0723_r2plus1d_ucf101_wo_label_circle'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        stem=dict(stem_type='2.5d'),
    ),
)
