import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,4,5"
# total_rounds = 540
# from ray import tune 

# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # The objective is to run the fine-tuning experiment by using the best hyper-parameters, i.e., 
# batch_size = 24/4= 6
# learing_rate = 0.01

# ucf-101-loss-based-yan,et.al
batch_size = 8
learning_rate = 0.01
##################################################################################################################################################################3
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/FedAvg_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained  /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# ucf-101-loss-based w/o momentum

# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/loss-based_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])


# #  ucf-101-FedU w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/FedU_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

#  ucf-101t-FedAvg_thea_b_only_w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/FedAvg_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_up_theta_b_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

#  ucf-101t-Loss_thea_b_only_w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir  /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/Loss_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_up_theta_b_loss_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])


#  ucf-101t-FedAvg+SWA_thea_b_only_w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/FedAvg+SWA_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1E_up_theta_b_only_FedAvg+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# #  ucf-101t-Loss+SWA_thea_b_only_w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/Loss+SWA_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1E_up_theta_b_only_loss+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

#  ucf-101t-yan_alpha0.9_beta_0_theta_b_only_w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/yan_alpha0.9_beta_0_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1E_up_theta_b_only_yan_alpha0.9_beta_0/alpha0.9_beta0_round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

#  ucf-101t-yan_alpha0.9_beta_1_theta_b_only_w/o momentum
process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_ucf101.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe/yan_alpha0.9_beta_1_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1E_up_theta_b_only_yan_alpha0.9_beta_1/alpha0.9_beta1_round-{540}-weights.array_copy.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])


#############################################################################################################################################

# hmdb-51-FedAvg w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmd/FedAvg_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# # hmdb-51-loss-based w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir  /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/loss-based_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss-based_wo_moment/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])

# # # hmdb-51-FedU w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir  /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/FedU_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_FedU_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])

# # # hmdb-51-FedAvg_theta_b_only_w/ow/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir  /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/FedAvg_theta_b_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_up_theta_b_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# # # hmdb-51-Losss_theta_b_only_w/ow/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/Loss_theta_b_only_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_up_theta_b_loss_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# # # hmdb-51-FedAvg+SWA_theta_b_only_w/ow/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/FedAvg+SWA_theta_b_only_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1E_up_theta_b_only_FedAvg+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# # # hmdb-51-Losss+SWA_theta_b_only_w/ow/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/Loss+SWA_theta_b_only_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1E_up_theta_b_only_loss+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])


#  hmdb-51-yan_alpha0.9_beta_0_theta_b_only_w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/yan_alpha0.9_beta_0_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1E_up_theta_b_only_yan_alpha0.9_beta_0/alpha0.9_beta0_round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])


#  hmdb-51-yan_alpha0.9_beta_1_theta_b_only_w/o momentum
process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
"/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/Linearprobe_hmdb51.py", "4",\
f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/Linearprobe_hmdb/yan_alpha0.9_beta_1_theta_b_only_wo_momentum/hmdb_{8}_{learning_rate}/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/K400_1E_up_theta_b_only_yan_alpha0.9_beta_1/alpha0.9_beta1_round-540-weights.array_copy.npz",\
f"--validate",\
f"--batch_size {batch_size}",\
f"--lr_rate {learning_rate}"])


####################################################################################################################################################################################################################

# # hmdb-51-loss-based-yan,et.al
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/loss_aggregation/hmdb_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based_yan/loss_1localE_round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])
##############################################################################################################################################################################333

# ucf-101-fedavg
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/fedavg_aggregation/ucf_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r/glb_epochs/round-540-weights.npy.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])


# # hmdb-51-fedavg
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/fedavg_aggregation/hmdb_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/glb_rounds_ctp_5c1e540r/glb_epochs/round-540-weights.npy.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

#########################################################################################################################################################################################

# # ucf-101-bbone_loss_cls_min
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/bbone_loss_cls_min/ucf_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_bbone_min_cls_loss_avg/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])


# # hmdb-51-bbone_loss_cls_min
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/bbone_loss_cls_min/hmdb_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_bbone_min_cls_loss_avg/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

##################################################################################################################################################################################

# ucf-101-DAPU

# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/DAPU/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])


# # hmdb-51-DAPU
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/DAPU/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+DPAU/round-540-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])










#  ucf-101-FedAvg w/o momentum

# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedAvg_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])










# #  ucf-101-loss+FedU w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/loss+FedU_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+FedU_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])


# # # hmdb-51-loss+FedU w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/loss+FedU_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based+FedU_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])



# #  ucf-101-FedAvg+SWA w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedAVG+SWA_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FEDAVG+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])


# # # hmdb-51-FedAvg+SWA w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FEDAVG+SWA_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FEDAVG+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])


# #  ucf-101-loss+SWA w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/loss+SWA_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])


# # # hmdb-51-loss+SWA w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/loss+SWA_wo_momentum/hmdb_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss+SWA_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])

# #  ucf-101t-FedAvg_thea_b_only_w/o momentum
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/FedAvg_theta_b_only_wo_momentum/ucf_{8}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_K400_1E_up_theta_b_wo_moment/round-{540}-weights.array.npz",\
# f"--validate",\
# f"--batch_size {8}",\
# f"--lr_rate {learning_rate}"])


























#####################################################################################################################################################################################3333
                                                                                    # centralized fine-tuning
#########################################################################################################################################################################################
# batch_size = 8  
# learning_rate = 0.01
# #centralized_20e_ucf
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/centralized/ucf_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_20.pth",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])

# # centralized_20e_hmdb
# process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
# "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_hmdb51.py", "4",\
# f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/strategy/centralized/hmdb_{batch_size}_{learning_rate}/",
# f"--data_dir /home/data3/DATA/",\
# f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/pretrain_27/epoch_20.pth",\
# f"--validate",\
# f"--batch_size {batch_size}",\
# f"--lr_rate {learning_rate}"])











