import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseRecognizer(nn.Module):
    """Base class for recognizers"""

    __metaclass__ = ABCMeta

    def __init__(self, train_cfg: dict, test_cfg: dict):
        super(BaseRecognizer, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._is_training = False

    @property
    def phase_cfg(self):
        if self._is_training:
            return self.train_cfg
        else:
            return self.test_cfg

    def train(self, mode=True):
        super(BaseRecognizer, self).train(mode)
        self._is_training = mode

    def eval(self):
        super(BaseRecognizer, self).eval()
        self._is_training = False

    def forward(self, *args, **kwargs):
        if self._is_training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, *args, **kwargs):
        pass
