_base_ = ['../../recognizers/_base_/model_r3d18_cam.py',
          '../../recognizers/_base_/runtime_ucf101_cam.py']
work_dir = './output/pot_v2/ft_ucf101/0717_r3d_ucf101_label'
model = dict(
    backbone=dict(
        pretrained='',
    ),
    cls_head=dict(num_classes=101),
)
