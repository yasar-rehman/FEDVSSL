import os
import torch
import numpy as np
import mmcv
from flwr.common import parameters_to_weights


work_dir = '/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_wo_moment_variance'
total_files = os.listdir(work_dir)
# takes the files name starting with round
rounds = 0
for items in os.listdir(work_dir):
    if items.endswith('.npz'):
        rounds += 1
std_bb_all = []
std_head_all = []
prog_bar = mmcv.ProgressBar(rounds)
for round_num in range(2, rounds):
    # find client chpts for round - 1
    checkpoints_paths = []
    for f in os.listdir(work_dir):
        if f.startswith('client'):
            for item in os.listdir(work_dir + '/{}'.format(f)):
                if item.startswith('round{}_weights.pth'.format(round_num - 1)):
                    checkpoints_paths.append(work_dir + '/{}/{}'.format(f, item))
    
    assert len(checkpoints_paths) == 5
    
    # all layers
    chpt = torch.load(checkpoints_paths[0])
    state_dict = chpt['state_dict']
    layer_names = list(state_dict.keys())
    layer_names_valid_bb = []
    layer_names_valid_head = []
    for layer_name in layer_names:
        if not (layer_name.endswith('num_batches_tracked') or layer_name.endswith('running_mean') or layer_name.endswith('running_var') or layer_name.startswith('head')):
            layer_names_valid_bb.append(layer_name)
        elif layer_name.startswith('head'):
            layer_names_valid_head.append(layer_name)
    
    # load global chpt for current round    
    params = np.load("{}/round-{}-weights.array.npz".format(work_dir, round_num), allow_pickle=True)
    params = parameters_to_weights(params['arr_0'].item())
    
    weights_diff_bb_list = []
    weights_diff_head_list = []
    for chpt in checkpoints_paths:
        chpt = torch.load(chpt)
        state_dict = chpt['state_dict']
        
        state_dict_global_bb = [v for k, v in zip(state_dict.keys(), params) if k in layer_names_valid_bb]
        state_dict_client_bb = [v.numpy() for k, v in state_dict.items() if k in layer_names_valid_bb]
        weight_diff_bb = [np.square(A.flatten() - B.flatten()) for A, B in zip(state_dict_global_bb, state_dict_client_bb)]
        weights_diff_bb = np.sum(np.concatenate(weight_diff_bb)) / len(weight_diff_bb)
        weights_diff_bb_list.append(weights_diff_bb)
        state_dict_global_head = [v for k, v in zip(state_dict.keys(), params) if k in layer_names_valid_head]
        state_dict_client_head = [v.numpy() for k, v in state_dict.items() if k in layer_names_valid_head]
        weight_diff_head = [np.square(A.flatten() - B.flatten()) for A, B in zip(state_dict_global_head, state_dict_client_head)]
        weights_diff_head = np.sum(np.concatenate(weight_diff_head)) / len(weight_diff_head)
        weights_diff_head_list.append(weights_diff_head)     
        
    std_bb = np.mean(weights_diff_bb_list)
    std_head = np.mean(weights_diff_head_list)
    # std_bb = np.std(weights_diff_bb_list)
    # std_head = np.std(weights_diff_head_list)
    std_bb_all.append(std_bb)
    std_head_all.append(std_head)
    print('finished round {}'.format(round_num))
    prog_bar.update()

save_path = '/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/weight_divergence_suml2.txt'

with open(save_path, 'w') as f:
    prog_bar = mmcv.ProgressBar(len(std_bb_all))
    assert len(std_bb_all) == len(std_head_all)
    for i in range(len(std_bb_all)): 
        f.write('{},{}\n'.format(str(std_bb_all[i]), str(std_head_all[i])))
        prog_bar.update()
