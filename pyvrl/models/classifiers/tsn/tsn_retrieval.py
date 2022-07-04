import torch
from torch import nn
from typing import Dict

# from cvlib.utils import obj_from_dict
from mmcv.runner.utils import obj_from_dict

from ..base_recognizer import BaseRecognizer
from ....builder import (MODELS, build_backbone, build_head)

# build_st_module,
# build_seg_consensus,

@MODELS.register_module()
class RetrievalTSN3D(nn.Module):

    def __init__(self,
                 backbone: dict,
                 compress: dict = None,
                 *args,
                 **kwargs):
        super(RetrievalTSN3D, self).__init__()
        self.backbone = build_backbone(backbone)
        if compress is not None:
            self.compress = obj_from_dict(compress, nn)
        else:
            self.compress = None

    def forward(self, imgs: torch.Tensor, *args, **kwargs):
        assert imgs.dim() == 6
        b, n, c, t, h, w = imgs.size()
        imgs = imgs.view(b * n, c, t, h, w)
        feats = self.backbone(imgs)
        if self.compress is not None:
            feats = self.compress(feats)
        feats = feats.view(b, n, -1)
        return feats

