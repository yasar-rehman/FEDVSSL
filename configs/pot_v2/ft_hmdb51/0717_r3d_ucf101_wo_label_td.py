_base_ = ['../../recognizers/_base_/model_r3d_18.py',
          '../../recognizers/_base_/runtime_hmdb51.py']
work_dir = './output/pot_v2/ft_hmdb51/0717_r3d_ucf101_wo_label_td'
model = dict(
    backbone=dict(
        pretrained='output/pot_v2/pretext/0717_r3d_ucf101_wo_label_td/epoch_300.pth',
    ),
    cls_head=dict(num_classes=51),
)
