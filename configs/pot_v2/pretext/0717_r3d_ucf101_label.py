_base_ = ['./_base_/model_r3d_18.py',
          './_base_/runtime_ucf101.py']
work_dir = './output/pot_v2/pretext/0717_r3d_ucf101_label'

model = dict(
    pot_head=dict(with_label_head=True)
)
