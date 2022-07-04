# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import logging
from torch import nn
from collections import OrderedDict
from mmcv.runner import load_state_dict
import re
from flwr.common import parameter
import numpy as np


class BaseBackbone(nn.Module):
    """ Base class for backbone network. """
    def __init__(self):
        super(BaseBackbone, self).__init__()

    def init_from_pretrained(self, pretrained, logger) -> None:
        """ Initialize model weights from a pretrained model. The pretrained
        model can be produced by some proxy task, e.g. CtP and SpeedNet, or
        by ImageNet inflated weights just like I3D.

        Since the checkpoint of a proxy task usually contains some other
        components besides the backbone weights, e.g., the prediction head in
        CtP, we will the prefix term 'backbone.' to match the exact necessary
        weights.

        Args:
            pretrained (str): pretrained model path. (something like *.pth)
            logger (logging.Logger): output logger

        Returns:
            None

        """
        logger.info(f"Loading pretrained backbone from {pretrained}")
        # print(self.state_dict().keys())
        if pretrained.endswith('.npz'):
        ##################################################################################################
            # following changes are made here
            params = np.load(pretrained, allow_pickle=True)
            if pretrained.endswith('.array_copy.npz'):
                params = params['arr_0']
                params_list = []
                for item in params:
                    if type(item) != np.ndarray:
                        item = np.array(item)
                    params_list.append(item)

                params = params_list     
            else:
                params = params['arr_0'].item()
                params = parameter.parameters_to_weights(params)
            # print(len(params), len(self.state_dict().keys()))
            params_dict = zip(self.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            # checkpoint = state_dict
            load_state_dict(self, state_dict, strict=False, logger=logger)
    
    ################################################################################################## 
        else:
            checkpoint = torch.load(pretrained)
        # get state_dict from checkpoint
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint

            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                raise RuntimeError(f'No state_dict found in '
                                f'checkpoint file {pretrained}')
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v
                            for k, v in checkpoint['state_dict'].items()}
            # strip prefix of backbone
            if any([s.startswith('backbone.') for s in state_dict.keys()]):
                # the following changes are done to make it run during downstream tasks
                state_dict = {k[9:]: v
                            for k, v in state_dict.items()
                            if k.startswith('backbone.')}
            load_state_dict(self, state_dict, strict=False, logger=logger)
            # for k, v in checkpoint['state_dict'].items()
        # 
