import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import os
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner.checkpoint import load_state_dict, get_state_dict, save_checkpoint
import re
import ray
import mmcv
# from custom_server import SaveModelStrategy
import time
import shutil
import time
import shutil
from flwr.common import parameter
import pdb # for debugging

# pylint: disable=no-member
# DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ): #-> Optional[fl.common.Weights]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # save weights
            print(f"round-{rnd}-weights...",)

            glb_dir = '../glb_rounds_vcop_5c3e180r/glb_epochs/'
            mmcv.mkdir_or_exist(os.path.abspath(glb_dir))
            np.savez(os.path.join(glb_dir,f"round-{rnd + 487}-weights.npy"), *weights)
        return weights



# Flower Client
class SslClient(fl.client.NumPyClient):
    """Flower client implementing video SSL w/ PyTorch."""

    def __init__(self, model, train_dataset, test_dataset, cfg, args, distributed, logger,videossl):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.cfg = cfg
        self.distributed = distributed
        self.logger = logger
        self.videossl=videossl
        # self.round = 1 
       

    def get_parameters(self) -> List[np.ndarray]:
        # Return local model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Update local model parameters from a list of NumPy ndarray  
        ##########################################################33
        # bad assignment; can cause problems
        # for i,param in enumerate(self.model.state_dict()):
        #     self.model.state_dict()[param] = parameters[i]
        #########################################################
        

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def get_properties(self, ins):
        return self.properties
        
    def fit(self, parameters, config):
        # Update local model w/ global parameters
        self.set_parameters(parameters)
        # Get hyperparameters from config
        # Train model on client-local data

        self.cfg.lr_config =  dict(policy='step', step=[30, 60]) 
        if self.args.checkpoint is not None:
            checkpoint = self.args.checkpoint  
        else:
            chk_name_list = [fn for fn in os.listdir(self.cfg.work_dir) if fn.endswith('.pth')]
            chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('epoch')]
            if chk_epoch_list:
                chk_epoch_list.sort()
                checkpoint = os.path.join(self.cfg.work_dir, f'epoch_{chk_epoch_list[-1]}.pth')
                self.cfg.checkpoint = checkpoint
                # replace the old model_state_dict with the new model_state-dict       
    
        self.videossl.train_model_cl(
            model = self.model,
            train_dataset = self.train_dataset,
            args = self.args,
            cfg = self.cfg,
            distributed = self.distributed,
            logger = self.logger
            )
        # Return updated model parameters to the server
        num_examples = len(self.train_dataset)  # TODO len(self.trainloader)
        return self.get_parameters(), num_examples, {}

    def evaluate(self, parameters, config):
        
        self.set_parameters(parameters)
        
        # print("saving each round ckpts")
       
        chk_name_list = [fn for fn in os.listdir(self.cfg.glb_dir) if fn.endswith('.pth')]
        rnd_ckpt_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('round')] # gives us the numbers 
        if rnd_ckpt_list:
            rnd_ckpt_list.sort()
            round_checkpoint_new = os.path.join(self.cfg.glb_dir, f'round_{rnd_ckpt_list[-1]+1}.pth')
            torch.save(checkpoint, round_checkpoint_new)
        else:
            round_checkpoint_new = os.path.join(self.cfg.glb_dir, f'round_{1}.pth')
            torch.save(checkpoint, round_checkpoint_new)

        print(round_checkpoint_new)
        
        result = 0

        print ("The checkpoint for the round is saved")
        return float(0), int(len(self.test_dataset)), {"accuracy": float(result)}

if __name__ == "__main__":

    pool_size = 100  # number of dataset partions (= number of total clients)
    client_resources = {"num_cpus": 2,"num_gpus": 1}  # each client will get allocated 1 CPUs

    initial_parameters = None 
    # initial_parameters=''
    # if initial_parameters.endswith('.npz'):
    #     ##################################################################################################
    #     # following changes are made here
    #     params = np.load(initial_parameters, allow_pickle=True)
    #     params = params['arr_0'].item()
    #     # params = parameter.parameters_to_weights(params)
    #     initial_parameters = params
    # else:
    #     initial_parameters=None
    ################################################################################################## 
    
    # configure the strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.05,
        min_fit_clients=5,
        min_available_clients=100,
        initial_parameters=initial_parameters)



    def main(cid: str):
    # Parse command line argument `cid` (client ID)

        import _init_paths
        import videossl
#        os.environ["CUDA_VISIBLE_DEVICES"] = cid
        cid_plus_one = str(int(cid)+1)
        args =   Namespace(
                        #   cfg='../reproduce_papers/configs/vcop_client'+cid_plus_one+'/vcop_run_config/vcop_runtime_config.py',
                          cfg='../reproduce_papers/configs/vcop/r3d_18_kinetics/pretraining_fed.py',
                          checkpoint=None, cid=int(cid), data_dir='/home/data3/DATA/', gpus=1, launcher='none',
                          local_rank=0, progress=False, resume_from=None, rounds=6, seed=7, validate=False,
                          work_dir='../fed_ssl_niid_vcop/ctp_5c3e180r/client'+cid_plus_one)
        print("Starting client", args.cid)
        cfg = Config.fromfile(args.cfg)
        cfg.data.train.data_source.ann_file = '.../DATA/Kinetics-processed/annotations/Kinetics-400_annotations/client_dist'+cid_plus_one+'.json'
        # cfg.data.train.data_source.ann_file = 'ucf101/annotations/train_split_'+cid_plus_one+'.json
        # set up the configuration
        distributed, logger = videossl.set_config_mmcv(args, cfg)

        # load the model
        model = videossl.load_model(args, cfg)

        # load the training data
        train_dataset = videossl.load_data(args, cfg)

        # load the test data
        test_dataset = videossl.load_test_data(args, cfg)
   
        # Initialize and start client
        return SslClient(model, train_dataset, test_dataset, cfg, args, distributed, logger,videossl)
    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=main,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=180,
        strategy=strategy,
        ray_init_args=ray_config,
    )