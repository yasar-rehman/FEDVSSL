# _base_ = './pretraining.py'

# if testing on action recognition uncomment the below lines
_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_ucf101.py']

# work_dir = './output/vcop/r3d_18_kinetics/finetune_ucf101/'

model = dict(
    backbone=dict(
        pretrained=''
    )
)

#pretrained='./output/vcop/r3d_18_kinetics/pretraining/epoch_90.pth',
