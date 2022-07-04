import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4,5"
# total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#VCOP Centralized
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/vcop/r3d_18_kinetics/pretraining.py", "6",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/pretrain/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained None"])


# # Speed Centralised
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/speednet/r3d_18_kinetics/pretraining.py", "6",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_speed/pretrain/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/centralized_speednet/pretrain/epoch_90.pth",\
# f"--validate" ])



# Ctp federated pretraining
process_obj = subprocess.run(["python", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/fedssl/main_fed_32bpc.py"])


# Strategies w/o momentum





