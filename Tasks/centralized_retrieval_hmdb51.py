import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
# total_rounds = 540

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Centralized_Ctp
process_obj = subprocess.run(["python","/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/retrieval_corrected/hmdb51/",\
f"--data_dir", f"/home/data3/DATA/",\
f"--dataset_name", f"hmdb51",\
f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_27.pth",\
f"--progress"])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Centralized_VCOP
process_obj = subprocess.run(["python","/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/vcop/r3d_18_kinetics/finetune_ucf101.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/retrieval_corrected/hmdb51/",\
f"--data_dir", f"/home/data3/DATA/",\
f"--dataset_name", f"hmdb51",\
f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/pretrain/epoch_90.pth",\
f"--progress"])


# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Centralized_speed
process_obj = subprocess.run(["python","/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/speednet/r3d_18_kinetics/finetune_ucf101.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_speed/retrieval_corrected/hmdb51/",\
f"--data_dir", f"/home/data3/DATA/",\
f"--dataset_name", f"hmdb51",\
f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/centralized_speednet/pretrain/epoch_90.pth",\
f"--progress"])
