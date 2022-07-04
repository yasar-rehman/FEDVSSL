import os
import torch
import numpy as np
import mmcv


def MSE(state_dict1, state_dict2):
    weight_diff = [np.square(A.flatten() - B.flatten()) for A, B in zip(state_dict1, state_dict2)]
    weights_diff = np.sum(np.concatenate(weight_diff))/len(weight_diff)
    return weights_diff
    
    return weight_diff_bb, weight_diff_pt

work_dir = '/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_1localE_FedAvg_wo_moment_variance'
total_files = os.listdir(work_dir)
# takes the files name starting with round
rounds = 0
for items in os.listdir(work_dir):
    if items.endswith('.npz'):
        rounds += 1
prog_bar = mmcv.ProgressBar(rounds)
backbone_var_all_list = []
head_var_all_list = []
for round_num in range(1, rounds): # for all rounds
    checkpoints_paths = []
    for f in os.listdir(work_dir): # directory
        if f.startswith('client'): # clients
            # print(f'the client: {f} is loaded')
            for item in os.listdir(work_dir + '/{}'.format(f)): # clients directory
                if item.startswith('round{}_weights.pth'.format(round_num)): #find if there is a file name round
                    checkpoints_paths.append(work_dir + '/{}/{}'.format(f, item)) # add it to the checkpoint
    
    assert len(checkpoints_paths) == 5
    
    state_dict_list = []
    for chpt in checkpoints_paths:
        chpt = torch.load(chpt)
        state_dict = chpt['state_dict']
        state_dict_list.append(state_dict)
    
#     all layers
    layer_names = list(state_dict_list[0].keys())
    layer_names_valid = []
    for layer_name in layer_names:
        layer_names_valid.append(layer_name)
        # if not (layer_name.endswith('num_batches_tracked') or layer_name.endswith('running_mean') or layer_name.endswith('running_var')):
        #     layer_names_valid.append(layer_name)
    
    # # last layer only
    # layer_names_valid = ['backbone.layer4.1.bn2.weight', 'backbone.layer4.1.bn2.bias', 'head.pred_head.weight', 'head.pred_head.bias']
    
    backbone_var_list = []
    head_var_list = []
    backbone_num_list = []
    head_num_list = []
    for layer_name in layer_names_valid:
        weights_list = []
        for state_dict in state_dict_list:
            weights_list.append(state_dict[layer_name].numpy())
        # var = np.var(weights_list, axis=0)
        var = np.std(weights_list, axis=0)
        var_mean = np.mean(var)
        
        weight_num = 1
        for d in var_mean.shape:
            weight_num *= d
        
        if layer_name.startswith('backbone'):
            backbone_var_list.append(var_mean)
            backbone_num_list.append(weight_num)
        else:
            head_var_list.append(var_mean)
            head_num_list.append(weight_num)
            
    backbone_var_all = sum([var * num for var, num in zip(backbone_var_list, backbone_num_list)]) / sum(backbone_num_list)      
    head_var_all = sum([var * num for var, num in zip(head_var_list, head_num_list)]) / sum(head_num_list)
    backbone_var_all_list.append(backbone_var_all)
    head_var_all_list.append(head_var_all)
    print(len(backbone_var_all_list), len(head_var_all_list))
    print('finished round {}'.format(round_num))
    prog_bar.update()
    
save_path = '/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/weight_diff_with_bn.txt'

with open(save_path, 'w') as f:
    prog_bar = mmcv.ProgressBar(len(backbone_var_all_list))
    assert len(backbone_var_all_list) == len(head_var_all_list)
    for i in range(len(backbone_var_all_list)): 
        f.write('{},{}\n'.format(str(backbone_var_all_list[i]), str(head_var_all_list[i])))
        prog_bar.update()
