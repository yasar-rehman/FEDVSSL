
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,4,5"

batch_size = 8
learning_rate = 0.01


# ctp_5c3e180r ucf

process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/ctp_5c3e180r/ucf_{8}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/data0/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c3e180r/glb_epochs/round-180-weights.npy.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])

# ctp_5c3e180r hmdb
process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/ctp_5c3e180r/hmdb_{8}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/data0/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c3e180r/glb_epochs/round-180-weights.npy.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])

