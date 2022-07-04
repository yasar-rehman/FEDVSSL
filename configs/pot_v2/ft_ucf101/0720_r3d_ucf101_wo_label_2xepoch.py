_base_ = ['../../recognizers/_base_/model_r3d_18.py',
          '../../recognizers/_base_/runtime_ucf101.py']
work_dir = './output/pot_v2/ft_ucf101/0720_r3d_ucf101_wo_label_2xepoch'
model = dict(
    backbone=dict(
        pretrained='output/pot_v2/pretext/0720_r3d_ucf101_wo_label_2xepoch/epoch_300.pth',
    ),
    cls_head=dict(num_classes=101),
)
