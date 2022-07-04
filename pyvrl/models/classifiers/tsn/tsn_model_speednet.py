# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from .tsn_modules import SimpleClsHead, SimpleSTModule
from ...train_step_mixin import TrainStepMixin
from ....builder import MODELS, build_backbone


@MODELS.register_module()
class TSN_speednet(nn.Module, TrainStepMixin):
    """ TSN action recognition model.
    Mainly ported from https://github.com/open-mmlab/mmaction project.
    """

    def __init__(self,
                 backbone: dict,
                 st_module: dict,
                 cls_head: dict):
        super(TSN_speednet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.st_module = SimpleSTModule(**st_module)
        self.cls_head = SimpleClsHead(**cls_head)
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if hasattr(self, 'st_module'):
            self.st_module.init_weights()
        if hasattr(self, 'cls_head'):
            self.cls_head.init_weights()

    def forward(self, return_loss=True, *args, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def _forward(self, imgs: torch.Tensor):
        """ Predict the classification results of the given video clip.
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
        Returns:
            cls_logits (torch.Tensor): classification results,
                in shape of [N, M, num_class]
        """
        batch_size = imgs.size(0)
        num_segs = imgs.size(1)
        # unsqueeze the first dimension
        imgs = imgs.view((-1, ) + imgs.shape[2:])
        # backbone network
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        feats = self.backbone(imgs)  # [NM, C, T, H, W]
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if self.st_module is not None:
            feats = self.st_module(feats)  # [NM, C, 1, 1, 1]
        cls_logits = self.cls_head(feats)
        cls_logits = cls_logits.view(batch_size, num_segs, -1)
        return cls_logits

    def forward_train(self,
                      imgs: torch.Tensor,
                      gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Forward 3D-Net and then return the losses
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
            gt_labels (torch.Tensor): ground-truth label in shape of [N, 1]
        """
        cls_logits = self._forward(imgs)
        gt_labels = gt_labels.view(-1)
        losses = self.cls_head.loss(cls_logits, gt_labels)
        return losses

    def forward_test(self, imgs: torch.Tensor, gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Forward 3D-Net and then return the classification results

        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]

        Returns:
            cls_scores (np.ndarray): in shape of [N, num_cls]

        """
        with torch.no_grad():
            losses = dict()
            cls_logits = self._forward(imgs)
            # average the classification logit
            cls_logits = cls_logits.mean(dim=1)
            # batch_size = cls_logits.size(0)
            ####################################################3
            # new items added here and modified some of them
            gt_labels = gt_labels.view(-1)
            losses = self.cls_head.loss(cls_logits, gt_labels)
            # cls_scores = torch.nn.functional.softmax(cls_logits, dim=1)
            # cls_preds = cls_scores.argmax(dim=1)
            # cls_scores = cls_scores.cpu().numpy()
            # print(gt_labels, cls_preds)
            # losses['accuracy'] = (cls_preds.eq(gt_labels.view(-1))).float().sum()
             
            
            # 
            # 
            #########################################################3
        return losses['acc'] 
