import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4"
# total_rounds = 540
# from ray import tune 

# # #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # The objective is to run the fine-tuning experiment by using the following hyper-parameters, learning-rate, vs batch-size

    # initially, we will load the configuration file first for modifying the learning rate and batch-size that will be 
    # Ctp Centralized_fine_tune_ucf
lr = [1e-1, 1e-2, 1e-3]
b_size = [2, 4, 6, 8]

for batch_size in b_size:
    for learning_rate in lr:
        process_obj = subprocess.run(["bash", "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/tools/dist_train.sh",\
        "/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
        f"--work_dir /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/centralized_ctp/centralized_finetune/ucf_{batch_size}_{learning_rate}/",
        f"--data_dir /home/data3/DATA/",\
        f"--pretrained /home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_loss_based_yan/loss_1localE_round-540-weights.array.npz",\
        f"--validate",\
        f"--batch_size {batch_size}",\
        f"--lr_rate {learning_rate}"])

    # after the training has finished we will use the work_dir directory to fetch the validation_loss configuration file and send the validation accuracy back to the tune







