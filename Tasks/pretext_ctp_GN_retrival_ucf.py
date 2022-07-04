import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
# total_rounds = 540

# Ctp federated with batch normalization

# FedAvg ctp_GN_UCF-101
process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101_GN.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ctp_retrieval/ctp_GN_2G_5c1e540r/",\
f"--data_dir", f"/home/data3/DATA",\
f"--dataset_name", f"ucf101",\
f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/reproduce_papers/glb_rounds_ctp_5c1e540r/glb_epochs_GroupNorm_2groups/round-540-weights.npy.npz",\
f"--progress"])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FedAvg_ctp_GN_HMDB-51
process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101_GN.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ctp_retrieval/ctp_GN_2G_5c1e540r/",\
f"--data_dir", f"/home/data3/DATA",\
f"--dataset_name", f"hmdb51",\
f"--checkpoint", f"home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/reproduce_papers/glb_rounds_ctp_5c1e540r/glb_epochs_GroupNorm_2groups/round-540-weights.npy.npz",\
f"--progress"])