import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4,5"
# total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#VCOP Centralized_fine_tune_hmdb51
process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/vcop/r3d_18_kinetics/finetune_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/centralized_finetune/hmdb51/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/pretrain/epoch_90.pth",\
f"--validate"])

# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Speed Centralised_fine_tune_hmdb51
process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/speednet/r3d_18_kinetics/finetune_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_speed/centralized_finetune/hmdb51/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/centralized_speednet/pretrain/epoch_90.pth",\
f"--validate" ])

# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ctp Centralized_fine_tune_hmdb51
process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/centralized_finetune/hmdb51/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_90.pth",\
f"--validate"])





