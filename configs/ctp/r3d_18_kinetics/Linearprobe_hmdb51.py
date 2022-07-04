_base_ = ['../../recognizers/_base_/model_r3d18_freeze.py',
          '../../recognizers/_base_/runtime_hmdb51_linear.py']

# work_dir = './output/ctp/r3d_18_kinetics/finetune_hmdb51/'

model = dict(
    backbone=dict(
        pretrained='',
    ),
    cls_head=dict(
        num_classes=51
    )
)