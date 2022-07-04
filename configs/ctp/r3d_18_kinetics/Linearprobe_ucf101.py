_base_ = ['../../recognizers/_base_/model_r3d18_freeze.py',
          '../../recognizers/_base_/runtime_ucf101_linear.py']

# work_dir = './output/ctp/r3d_18_kinetics/finetune_ucf101/'

model = dict(
    backbone=dict(
        pretrained='',
    ),
)
