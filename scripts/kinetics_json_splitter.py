import json
import random
import copy
import os
# Use this script file to create a data distribution  for federated learning from a .json file.  
json_path = '/home/data3/DATA/Kinetics-processed/annotations/train_in_official.json'
output_path = '/home/data3/DATA/Kinetics-processed/annotations/annotations_fed_iid'
random.seed(7)
num_clients = 100
import mmcv

if not os.path.isdir(output_path):
    mmcv.mkdir_or_exist(output_path)
else:
    pass


with open(json_path, 'r') as f:
    json_object = json.load(f)
    json_object_copy = copy.deepcopy(json_object)
    random.shuffle(json_object_copy)
    num_data = len(json_object_copy)
    for i in range(0,num_clients):
        client_data = json_object_copy[
                                        int(i* num_data / num_clients):
                                        int((i+1)* num_data / num_clients)
                                      ]
        with open(os.path.join(output_path,"client_dist" + str(i+1) + ".json"), "w") as f:
            json.dump(client_data, f, indent=2)
            f.close()
