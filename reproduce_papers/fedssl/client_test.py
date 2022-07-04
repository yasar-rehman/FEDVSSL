# from fedssl import __version__
import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import os
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
import _init_paths
import videossl
from mmcv.runner.checkpoint import load_state_dict

# def test_version():
#     assert __version__ == "0.1.0"

class test_client(object):
    def __init__(self, model, test_dataset, cfg, args, distributed, logger):
        self.model = model
        self.test_dataset = test_dataset
        self.args = args
        self.cfg = cfg
        self.distributed = distributed
        self.logger = logger

    def evaluate(self):
        result = videossl.test_model_cl(
                        model = self.model,
                        test_dataset = self.test_dataset,
                        args = self.args,
                        cfg = self.cfg,
                        distributed = self.distributed,
                        logger = self.logger
                        )
        return result


def main():
    
    # Parse command line argument `cid` (client ID)
    parser = argparse.ArgumentParser(description="Flower_test")
    
    

    # ------------------------------------------------------------------------------
    # ------------------------- MMcv arguments--------------------------------------
    # ------------------------------------------------------------------------------

    parser.add_argument('--cfg',
                        default='',
                        type=str, help='train config file path *.py')
    parser.add_argument('--work_dir',
                        default = '',
                        help='the dir to save logs and models.'
                             'if not specified, program will use the path '
                             'defined in the configuration file.')
    parser.add_argument('--data_dir',
                        default='/home/data3/DATA',
                        type=str,
                        help='the dir that save training data.'
                             '(data/ by default)')
    parser.add_argument('--resume_from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--validate',
                        action='store_true',
                        help='whether to evaluate the checkpoint during '
                             'training')
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='number of gpus to use '
                             '(only applicable to non-distributed training)')

    parser.add_argument('--checkpoint',
                        default='', 
                        help='checkpoint path')

    parser.add_argument('--progress', 
                        action='store_true', 
                        help='show progress bar')
    parser.add_argument('--seed', type=int, default=7, help='random seed')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    args = parser.parse_args()
    print(args)
    cfg = Config.fromfile(args.cfg)

    # set up the configuration
    distributed, logger = videossl.set_config_mmcv(args, cfg)

    # load the model
    model = videossl.load_model(args, cfg)

    # load the test data
    test_dataset = videossl.load_test_data(args, cfg)

    
    test_obj = test_client(model, test_dataset, cfg, args, distributed, logger)
    test_obj.evaluate()


if __name__ == "__main__":
    main()