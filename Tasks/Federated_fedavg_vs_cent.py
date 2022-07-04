import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,4,5"
total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# centralized_ctp_ucf
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_fed_vs_cent.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp_vs_fedavg/ctp/hmdb51_180r_90e/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--ran_multip", f"{0}",\
# f"--progress"])

# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_fed_vs_cent.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp_vs_fedavg/ctp/ucf101_180r_90e/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--ran_multip", f"{0}",\
# f"--progress"])



# # centralized_ctp_ucf
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_fed_vs_cent.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp_vs_fedavg/ctp/hmdb51_180r/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--ran_multip", f"{0}",\
# f"--progress"])


process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_fed_vs_cent.py",\
f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp_vs_fedavg/ctp/ucf101/",\
f"--data_dir", f"/home/data3/DATA/",\
f"--dataset_name", f"ucf101",\
f"--ran_multip", f"{0}",\
f"--progress"])





