import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional
import os
import flwr as fl
import numpy as np
from math import exp
import torch
import torch.nn as nn
import mmcv
from mmcv import Config
from mmcv.runner.checkpoint import load_state_dict, get_state_dict, save_checkpoint
import re
import ray
import time
import shutil
from flwr.common import parameter
from functools import reduce
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

# pylint: disable=no-member
# DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

DIR = '1localE_loss+vect_add'


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.metrics['loss'])
            for client, fit_res in results
        ]

        weights = aggregate(weights_results)

        weights = weights_to_parameters(weights)

        if weights is not None:
            # save weights
            print(f"round-{rnd}-weights...",)

            glb_dir = '/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_' + DIR
            mmcv.mkdir_or_exist(os.path.abspath(glb_dir))
            np.savez(os.path.join(glb_dir, f"round-{rnd}-weights.array"), weights)

        return weights, {}


def aggregate(results: List[Tuple[Weights, float]]) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

# order classes by number of samples
def takeSecond(elem):
    return elem[1]


# Flower Client
class SslClient(fl.client.NumPyClient):
    """Flower client implementing video SSL w/ PyTorch."""

    def __init__(self, model, train_dataset, test_dataset, cfg, args, distributed, logger, videossl):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.cfg = cfg
        self.distributed = distributed
        self.logger = logger
        self.videossl = videossl

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
        # the state_dict_cr only collect the values of the current model 
        state_dict_cr = [v.cpu().numpy() for _, v in self.model.state_dict().items()]
        
        # Average the two weights
        weights_diff = [np.asarray((0.99*B + 0.01*A)) for A, B in zip(state_dict_cr, parameters)]


        params_dict = zip(self.model.state_dict().keys(), weights_diff)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    
    def get_properties(self, ins):
        return self.properties

    def fit(self, parameters, config):
        # Update local model w/ global parameters
        self.set_parameters(parameters)
        # Get hyperparameters from config
        # Train model on client-local data

        # change lr based on global round
        global_round = int(config["epoch_global"])

        self.cfg.lr_config = dict(policy='step', step=[100, 200])

        self.videossl.train_model_cl(
            model=self.model,
            train_dataset=self.train_dataset,
            args=self.args,
            cfg=self.cfg,
            distributed=self.distributed,
            logger=self.logger
        )
        # Return updated model parameters to the server
        num_examples = len(self.train_dataset)  # TODO len(self.trainloader)

        # fetch loss from log file
        work_dir = self.args.work_dir
        log_f_list = []
        for f in os.listdir(work_dir):
            if f.endswith('log.json'):
                num = int(''.join(f.split('.')[0].split('_')))
                log_f_list.append((f, num))

        # take the last log file
        log_f_list.sort(key=takeSecond)
        log_f_name = work_dir + '/' + log_f_list[-1][0]
        loss_list = []
        with open(log_f_name, 'r') as f:
            for line in f.readlines():
                line_dict = eval(line.strip())
                loss = float(line_dict['loss'])
                loss_list.append(loss)

        avg_loss = sum(loss_list) / len(loss_list)
        exp_loss = exp(- avg_loss)
        metrics = {'loss': exp_loss}

        return self.get_parameters(), num_examples, metrics

    def evaluate(self, parameters, config):
       
        result = 0
        
        return float(0), int(len(self.test_dataset)), {"accuracy": float(result)}


def initial_setup(cid, base_work_dir, rounds, light=False):
    import _init_paths
    import videossl
    cid_plus_one = str(int(cid) + 1)
    args = Namespace(
        cfg='../reproduce_papers/configs/ctp/r3d_18_kinetics/pretraining_fed.py',
        checkpoint=None, cid=int(cid), data_dir='/home/data3/DATA/', gpus=1,
        launcher='none',  # '/nfs-share/pedro/kinetics_processed/400' or '/hdd1/datasets/'
        local_rank=0, progress=False, resume_from=None, rounds=6, seed=7, validate=False,
        work_dir=base_work_dir + '/client' + cid_plus_one)
    
    print("Starting client", args.cid)
    cfg = Config.fromfile(args.cfg)
    # cfg.total_epochs = 1  ### Used for debugging. Comment to let config set number of epochs
    cfg.data.train.data_source.ann_file = 'Kinetics-processed/annotations/Kinetics-400_annotations/client_dist' + cid_plus_one + '.json'
    # cfg.data.val.data_source.ann_file = 'val/val_in_official_clean_shrunk.json'
    # set up the configuration
    if light:
        distributed, logger = videossl.set_config_mmcv_light(args, cfg)
    else:
        distributed, logger = videossl.set_config_mmcv(args, cfg)
    # load the model
    model = videossl.load_model(args, cfg)
    # load the training data
    train_dataset = videossl.load_data(args, cfg)
    # load the test data
    test_dataset = videossl.load_test_data(args, cfg)
    return args, cfg, distributed, logger, model, train_dataset, test_dataset, videossl


def _temp_get_parameters(model):
    # Return local model parameters as a list of NumPy ndarrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
    }
    return config


if __name__ == "__main__":
    import _init_paths
    import videossl

    os.chdir("/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/fedssl/")
    pool_size = 100  # number of dataset partions (= number of total clients)
    client_resources = {"num_cpus": 2, "num_gpus": 1}  # each client will get allocated 1 CPUs
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    base_work_dir = '/home/root/yasar/SSFVRL/federated-unsupervised-learning/videoSSL/reproduce_papers/k400_' + DIR
    rounds = 540


    # configure the strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.05,
        fraction_eval=0.02,
        min_fit_clients=5,
        min_eval_clients=1,
        min_available_clients=pool_size,
        on_fit_config_fn=fit_config,
        # initial_parameters=parameters_init,
        # eval_fn=central_evaluator(base_work_dir, rounds,pool_size),
    )


    def main(cid: str):
        # Parse command line argument `cid` (client ID)
        #        os.environ["CUDA_VISIBLE_DEVICES"] = cid
        args, cfg, distributed, logger, model, train_dataset, test_dataset, videossl = initial_setup(cid, base_work_dir,
                                                                                                     rounds)
        return SslClient(model, train_dataset, test_dataset, cfg, args, distributed, logger, videossl)


    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=main,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )
