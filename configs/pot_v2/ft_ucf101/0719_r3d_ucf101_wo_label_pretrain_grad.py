_base_ = ['../../recognizers/_base_/model_r3d_18.py',
          '../../recognizers/_base_/runtime_ucf101.py']
work_dir = './output/pot_v2/ft_ucf101/0719_r3d_ucf101_wo_label_pretrain_grad'
model = dict(
    backbone=dict(
        pretrained='output/pot_v2/pretext/0719_r3d_ucf101_wo_label_pretrain_grad/epoch_300.pth',
    ),
    cls_head=dict(num_classes=101),
)
