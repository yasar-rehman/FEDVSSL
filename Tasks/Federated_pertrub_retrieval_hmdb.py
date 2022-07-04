import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
total_rounds = 540


# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ctp _perturbation
perturbation_multipler = [0.1, 0.2, 0.3, 0.4, 0.5]

for i in perturbation_multipler:
    # centralized_ctp
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Perturbation_experiment/centralized_ctp/ctp_perturbed_ver_sigma_{i}/hmdb51/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_90.pth",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # federated_ctp_5c1e540r

    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Perturbation_experiment/Federated_ctp/ctp_perturbed_{i}/hmdb51/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r/glb_epochs/round-540-weights.npy.npz",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # # federated_ctp_5c3e180r
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_ctp/ctp_perturbed/hmdb51_5c3e180r_{i}/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c3e180r/glb_epochs/round-180-weights.npy.npz",\
    # f"--ran_multip", f"{i}",\
    # f"--progress"])

    # centralized_speed
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Perturbation_experiment/centralized_speed/speed_perturbed/hmdb51_centralized_{i}/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/centralized_speednet/pretrain/epoch_90.pth",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # Federated_speed_5c1e540r
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Perturbation_experiment/Federated_speed/speed_perturbed/hmdb51_5c1e540r_{i}/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_speednet_5c1e540r_extra/glb_epochs/round-540-weights.npy.npz",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # # Federated_speed_5c3e180r
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_speed/speed_perturbed/hmdb51_5c1e180r_{i}/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_speednet_5c3e180r/glb_epochs/round-180-weights.npy.npz",\
    # f"--ran_multip", f"{i}",\
    # f"--progress"])

    # centralized_vcop
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Perturbation_experiment/centralized_vcop/vcop_perturbed/hmdb51_centralized_{i}/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_vcop/pretrain/epoch_90.pth",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # Federated_vcop_5c1e540r
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Perturbation_experiment/Federated_vcop/vcop_perturbed/hmdb51_5c1e540r_{i}/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_vcop/glb_epochs/round-540-weights.npy.npz",\
    f"--ran_multip", f"{i}",\
    f"--progress"])

    # # Federated_vcop_5c3e180r
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2_perturbation.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Federated_vcop/vcop_perturbed/hmdb51_5c3e180r_{i}/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_vcop_5c3e180r_yan/aggregated_model_180.pth",\
    # f"--ran_multip", f"{i}",\
    # f"--progress"])


