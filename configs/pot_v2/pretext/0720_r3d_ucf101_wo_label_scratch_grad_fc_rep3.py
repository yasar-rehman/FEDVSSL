_base_ = ['./0720_r3d_ucf101_wo_label_scratch_grad_fc_rep.py']
work_dir = './output/pot_v2/pretext/0720_r3d_ucf101_wo_label_scratch_grad_fc_rep3'

model = dict(
    pot_head=dict(
        with_label_head=False,
        roi_grad=True,
        with_bn=False
    ),
    force_circle=True,
    force_v1=False,
)
