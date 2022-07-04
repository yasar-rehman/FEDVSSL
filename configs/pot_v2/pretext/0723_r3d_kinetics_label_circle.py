_base_ = ['./0723_r3d_ucf101_label_circle.py']
work_dir = './output/pot_v2/pretext/0723_r3d_kinetics_label_circle'

data = dict(train=dict(name='kinetics400_train'))

# optimizer
total_epochs = 90
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[40, 70]
)
