import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,4,5"
# total_rounds = 540

batch_size = 8
learning_rate = 0.01

# FedU_UCF
# # process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedU/ucf_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# FedU_HMDB
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedU/hmdb_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])
##########################################################################################################################################################3333

# Loss-based+FedU_UCF
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Lossbased+FedU/ucf_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU_w_local_state/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# Loss-based+FedU_HMDB
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Loss-based+FedU/hmdb_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU_w_local_state/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

############################################################################################################################################################################


# Loss-based+SWA_UCF
process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/loss-based+SWA/ucf_{batch_size}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based+SWA/round-540-weights.array.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])

# Loss-based+SWA_HMDB
process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/loss-based+SWA/hmdb_{batch_size}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based+SWA/round-540-weights.array.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])

############################################################################################################################################################################

# FedAvg+SWA_UCF
process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedAvg+SWA/ucf_{batch_size}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1localE_FedAvg+SWA/round-540-weights.array.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])

# FedAvg+SWA_HMDB
process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedAvg+SWA/hmdb_{batch_size}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1localE_FedAvg+SWA/round-540-weights.array.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])

############################################################################################################################################################################





#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ctp_5c1e540r
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_avg_cls_loss/retrieval_5c1e540r/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss/round-540-weights.array.npz",\
# f"--progress"])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# fed_bbone_loss_cls_avg_ucf
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_loss_cls_avg/retrieval_5c1e540r/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_bbone_loss_cls_avg/round-540-weights.array.npz",\
# f"--progress"])
# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # fed_bbone_loss_cls_avg_hmdb
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_loss_cls_avg/retrieval_5c1e540r/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_bbone_loss_cls_avg/round-540-weights.array.npz",\
# f"--progress"])
# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# fed_loss-based_vcop_yan_ucf
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_vcop_loss-based/retrieval_5c1e540r/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based_yan/vcop/vcop_loss_1localE_round-69-weights.array.npz",\
# f"--progress"])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# fed_bbone_min_clsfedavg
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_min_cls_avg/retrieval_5c1e540r/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_bbone_min_cls_avg/round-540-weights.array.npz",\
# f"--progress"])
# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # fed_bbone_min_clsfedavg
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_min_cls_avg/retrieval_5c1e540r/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_bbone_min_cls_avg/round-540-weights.array.npz",\
# f"--progress"])
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# fed_bbone_min_cls-loss
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_min_cls_loss/retrieval_5c1e540r/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_bbone_min_cls_loss_avg/round-540-weights.array.npz",\
# f"--progress"])
# # #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # # fed_bbone_min_cls-loss
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_min_cls_loss/retrieval_5c1e540r/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_bbone_min_cls_loss_avg/round-540-weights.array.npz",\
# f"--progress"])

# fed_bbone_loss_cls-DPAU
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_loss_cls_DPAU/retrieval_5c1e540r/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU/round-540-weights.array.npz",\
# f"--progress"])
# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # fed_bbone_loss_cls-DPAU
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_bbone_loss_cls_DPAU/retrieval_5c1e540r/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU/round-540-weights.array.npz",\
# f"--progress"])


# # fed_bbone_loss-based+FedU
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_loss-based+FEDU/retrieval_5c1e540r/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU_w_local_state/round-540-weights.array.npz",\
# f"--progress"])
# # #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # # fed_bbone_loss-based+FedU
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/fed_ctp_loss-based+FedU/retrieval_5c1e540r/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU_w_local_state/round-540-weights.array.npz",\
# f"--progress"])


# Ctp_FedU
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Ctp_FedU/retrieval_5c1e540r/ucf/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"ucf101",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU/round-540-weights.array.npz",\
# f"--progress"])
# # # #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # # Ctp_FedU
# process_obj = subprocess.run(["python","/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/test_clip_retrieval2.py",\
# f"--cfg", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py",\
# f"--work_dir", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/Ctp_FedU/retrieval_5c1e540r/hmdb/",\
# f"--data_dir", f"/home/data3/DATA/",\
# f"--dataset_name", f"hmdb51",\
# f"--checkpoint", f"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU/round-540-weights.array.npz",\
# f"--progress"])


