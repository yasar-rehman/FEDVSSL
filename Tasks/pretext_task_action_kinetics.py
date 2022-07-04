import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4,5"
# total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#VCOP Centralized
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/vcop/r3d_18_kinetics/pretraining.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/pretrain/",
# f"--data_dir /home/data3/DATA/"])
# # f"--pretrained None"])


# # Speed Centralised
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/speednet/r3d_18_kinetics/pretraining.py", "6",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_speed/pretrain/",
# f"--data_dir /home/data3/DATA/"])



# Ctp Centralized pretraining
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/pretraining.py", "6",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_101/",
# f"--data_dir /home/data3/DATA/"])
# # f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_27.pth"])

# Moco Centralized pretraining
process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/moco/r3d_18_kinetics/pretraining.py", "6",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_moco/pretrain_10/",
f"--data_dir /home/data3/DATA/"])


# Mem DPC pretraining
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/memdpc/r3d_18_kinetics/pretraining.py", "6",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_memdpc/pretrain_10/",
# f"--data_dir /home/data3/DATA/"])


