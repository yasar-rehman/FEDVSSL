import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
total_rounds = 540

# ctp_loss-based w/o momentum
# rounds = [1, 100, 200, 300, 400, 500, 540]
# for i in round:
process_obj = subprocess.run(["python", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/grad_cam.py",\
f"--cfg", f"/home/root/yasar/reproduce_papers/configs/pot_v2/ft_ucf101/0717_r3d_ucf101_label.py",\
f"--data_dir", f"/home/data3/DATA/",\
f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/finetune_linear_vssl/action_ucf_pretext/Speed_5c1e540r/dtask_ucf_32/epoch_100.pth"])
# print(f"the {i} process has finished")
