import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,4,5"
total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ctp _perturbation
perturbation_multipler = [0.1, 0.2, 0.3, 0.4, 0.5]

for i in perturbation_multipler:
    # centralized_ctp_ucf
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp_27/ctp_perturbed_ver_sigma_{i}/ucf/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"ucf101",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_27.pth",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # centralized_ctp_hmdb51
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp_27/ctp_perturbed_ver_sigma_{i}/hmdb51/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_27.pth",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # # federated_ctp_iid
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_ctp/ctp_IID_perturbed_{i}/ucf/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e90r_iid/first_try/round-540-weights.npy.npz",\
    # f"--ran_multip", f"{i}",\
    # f"--progress"])

    # # federated_ctp_iid_hmdb
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_ctp/ctp_IID_perturbed_{i}/hmdb51/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e90r_iid/first_try/round-540-weights.npy.npz",\
    # f"--ran_multip", f"{i}",\
    # f"--progress"])

    


