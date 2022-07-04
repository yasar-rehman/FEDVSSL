import torch
from typing import Dict
from ..base_recognizer import BaseRecognizer
from ....builder import (MODELS, build_backbone, build_head)
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from .tsn_modules import SimpleClsHead, SimpleSTModule, SimpleConsensus


@MODELS.register_module()
class RecognizerTSN3D(BaseRecognizer):

    def __init__(self,
                 backbone: dict,
                 st_module: dict,
                 seg_consensus: dict,
                 cls_head: dict,
                 train_cfg: dict,
                 test_cfg: dict):
        super(RecognizerTSN3D, self).__init__(train_cfg=train_cfg, test_cfg=test_cfg)
        self.backbone = build_backbone(backbone)
        self.st_module = SimpleSTModule(**st_module)
        self.seg_consensus = SimpleConsensus(**seg_consensus)
        self.cls_head = SimpleClsHead(**cls_head)
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if hasattr(self, 'st_module'):
            self.st_module.init_weights()
        if hasattr(self, 'seg_consensus'):
            self.seg_consensus.init_weights()
        if hasattr(self, 'cls_head'):
            self.cls_head.init_weights()

    def _forward(self, imgs: torch.Tensor):
        batch_size = imgs.size(0)
        num_segs = imgs.size(1)
        # unsqueeze the first dimension
        imgs = imgs.view((-1, ) + imgs.shape[2:])
        # backbone network
        feats = self.backbone(imgs)
        if self.st_module is not None:
            feats = self.st_module(feats)
        if self.seg_consensus is not None:
            feats = feats.reshape((-1, num_segs) + feats.shape[1:])
            feats = self.seg_consensus(feats)
            feats = feats.squeeze(1)
        cls_logits = self.cls_head(feats)
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

    def forward_test(self, imgs: torch.Tensor):
        """ Forward 3D-Net and then return the classification results
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
        """
        cls_logits = self._forward(imgs)
        cls_scores = torch.nn.functional.softmax(cls_logits, dim=1)
        return cls_scores

