import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,2,3,4,5"
# total_rounds = 540



# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/FedAvg_w_moment_ctp_10_clients/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_w_moment_10_clients/round-{540}-weights.array.npz",\
# f"--progress"])


# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/FedAvg_w_moment_ctp_10_clients/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_w_moment_10_clients/round-{540}-weights.array.npz",\
# f"--progress"])


# # # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# batch_size = 8
# learning_rate = 0.01

# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/FedAvg_w_momentum_10_clients/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_w_moment_10_clients/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])


# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/FedAvg_w_momentum_10_clients/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_w_moment_10_clients/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])
# ##############################################################################################################################################################

batch_size = 6
learning_rate = 0.01

# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedAvg_w_momentum_10_clients/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_w_moment_10_clients/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])

process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedAvg_w_momentum_10_clients/hmdb_{8}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_w_moment_10_clients/round-{540}-weights.array.npz",\
f"--validate",\
f"--batch_size {8}",\
f"--lr_rate {learning_rate}"])








