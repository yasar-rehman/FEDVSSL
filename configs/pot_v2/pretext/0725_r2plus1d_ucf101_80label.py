_base_ = './0724_r3d_ucf101_80lb_pos_nocls.py'
work_dir = './output/pot_v2/pretext/0725_r2plus1d_ucf101_80label'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        stem=dict(stem_type='2.5d'),
    ),
)
