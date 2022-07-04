import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "3,4,5"
total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ctp _perturbation
perturbation_multipler = [(0.3 + (x/100)) for x in range(10)]

for i in perturbation_multipler:
      # Federated_vcop_5c1e540r
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/vcop_perturbed/ucf_centralized_{i}/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"ucf101",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/pretrain/epoch_90.pth",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # # Federated_vcop_5c3e180r
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_vcop/vcop_perturbed/ucf_5c3e180r_{i}/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_vcop_5c3e180r_yan/aggregated_model_180.pth",\
    # f"--ran_multip", f"{i}",\
    # f"--progress"])


