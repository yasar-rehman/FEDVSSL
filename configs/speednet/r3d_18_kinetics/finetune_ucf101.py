_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_ucf101.py']

# work_dir = './output/speednet/r3d_18_kinetics/finetune_ucf101/'

model = dict(
    backbone=dict(
        pretrained='',
    ),
)
