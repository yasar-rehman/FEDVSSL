import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
# total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Ctp federated_retrieval_ucf
process_obj = subprocess.run(["python","/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_ctp/ctp_4cpc_5c1e540r_corrected/ucf/",\
f"--data_dir", f"/home/data3/DATA/",\
f"--dataset_name", f"ucf101",\
f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r_4cpc_niid/glb_epochs/round-540-weights.npy.npz",\
f"--progress"])

# Ctp federated_retrieval_hmdb51
process_obj=subprocess.run(["python","/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_ctp/ctp_4cpc_5c1e540r_corrected/hmdb51/",\
f"--data_dir", f"/home/data3/DATA/",\
f"--dataset_name", f"hmdb51",\
f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r_4cpc_niid/glb_epochs/round-540-weights.npy.npz",\
f"--progress"])

# # Ctp federated_fine_tune_ucf
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_ctp/ctp_4cpc_5c1e540r/ucf/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r_4cpc_niid/glb_epochs/round-540-weights.npy.npz",\
# f"--validate"])


# # Ctp federated_fine_tune_hmdb51
# process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_ctp/ctp_4cpc_5c1e540r/hmdb51/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r_4cpc_niid/glb_epochs/round-540-weights.npy.npz",\
# f"--validate"])





