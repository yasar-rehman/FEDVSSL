_base_ = ['./0717_r3d_ucf101_wo_label.py']
work_dir = './output/pot_v2/pretext/0717_r3d_kinetics_wo_label'

data = dict(
    train=dict(name='kinetics400_train'))

total_epochs = 90
lr_config = dict(
    policy='step',
    step=[40, 70]
)

