import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
# total_rounds = 540
rounds = [540]
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i in rounds:
    # Ctp_fedavg
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/fedavg/ctp_round_{i}/ucf/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/data0/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r/glb_epochs/round-{i}-weights.npy.npz",\
    # f"--progress"])

    # Ctp_Loss-based
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/loss-based/ucf/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based/round-{i}-weights.array.npz",\
    # f"--progress"])


    # Ctp_FedU
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/FedU/ctp/ucf/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU/round-{i}-weights.array.npz",\
    # f"--progress"])


    # # Ctp_loss-based+FedU
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/loss-based+FedU/ctp/ucf/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU_w_local_state/round-{i}-weights.array.npz",\
    # f"--progress"])

    # Ctp_FedAvg+SWA
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/Fedavg+SWA/ctp/ucf/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1localE_FedAvg+SWA/round-{i}-weights.array.npz",\
    # f"--progress"])

    # # Ctp_loss-based + SWA
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/loss-based+SWA/ctp/ucf/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"ucf101",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based+SWA/round-{i}-weights.array.npz",\
    # f"--progress"])


    ###################################################################################################################################################################
                                                                            # HMDB-51
    ###################################################################################################################################################################

    # Ctp_fedavg
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/fedavg/ctp/hmdb/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/data0/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r/glb_epochs/round-{i}-weights.npy.npz",\
    # f"--progress"])


    # # Ctp_FedU
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/FedU/ctp/hmdb/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU/round-{i}-weights.array.npz",\
    # f"--progress"])


    # # Ctp_loss-based+FedU
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/loss-based+FedU/ctp/hmdb/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU_w_local_state/round-{i}-weights.array.npz",\
    # f"--progress"])

    # # Ctp_FedAvg+SWA
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/Fedavg+SWA/ctp/hmdb/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1localE_FedAvg+SWA/round-{i}-weights.array.npz",\
    # f"--progress"])

    # # Ctp_loss-based + SWA
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/loss-based+SWA/ctp/hmdb/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based+SWA/round-{i}-weights.array.npz",\
    # f"--progress"])

    # ctp_loss-based
    # process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    # f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    # f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/loss-based/hmdb/",\
    # f"--data_dir", f"/home/data3/DATA/",\
    # f"--dataset_name", f"hmdb51",\
    # f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based/round-{i}-weights.array.npz",\
    # f"--progress"])

    ########################################################################################################################################################################
    
    # ctp_loss-based w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/loss_based_wo_momen/ucf/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"ucf101",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])

    # ctp_loss-based w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/loss_based_wo_momen/hmdb/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])
    ################################################################################################################################################################################

    # ctp_FedAvg w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/FedAvg_wo_moment/ucf/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"ucf101",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])

    # ctp_FedAvg w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/FedAvg_wo_moment/hmdb/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])


    # ctp_FedU w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/FedU_wo_moment/ucf/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"ucf101",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])

    # ctp_FedU w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/FedU_wo_moment/hmdb/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])

    # ctp_loss_FedU w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/loss_FedU_wo_moment/ucf/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"ucf101",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+FedU_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])

    # # ctp_loss_FedU w/o momentum
    process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
    f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
    f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy_visualize/loss_FedU_wo_moment/hmdb/",\
    f"--data_dir", f"/home/data3/DATA/",\
    f"--dataset_name", f"hmdb51",\
    f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+FedU_wo_moment/round-{i}-weights.array.npz",\
    f"--progress"])



###########################################################################################################################################################################
                                                                                # Ctp_Centralized
###########################################################################################################################################################################

# ctp_centralized_10epochs_ucf
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/centralized/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_20.pth",\
# f"--progress"])

# # ctp_centralized_10epochs_hmdb
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/ablation_strategy/centralized/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_20.pth",\
# f"--progress"])



