# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from typing import Union, List


def img_np2tensor(img: Union[np.ndarray, List[np.ndarray]], norm: bool = True):
    """ Convert numpy image to torch.Tensor. """
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
    elif isinstance(img, (list, tuple)):
        img = [np.expand_dims(i_img, axis=0) for i_img in img]
        img = np.concatenate(img, axis=0)
    else:
        raise TypeError("Unknown type {}".format(type(img)))

    assert img.ndim == 4
    img = torch.from_numpy(img).float()  # [N, H, W, 3]
    img = img.permute(0, 3, 1, 2).contiguous()
    if norm:
        img = img_norm(img)

    return img


def img_tensor2np(img: torch.Tensor, denorm: bool = True) -> List[np.ndarray]:
    if denorm:
        img = img_denorm(img)
    assert img.dim() == 4
    img = img.cpu()
    img = img.permute(0, 2, 3, 1).contiguous().numpy().astype(np.uint8)
    img_list = [img[i] for i in range(img.shape[0])]
    return img_list


def img_gray(img):
    """ convert to gray scale

    Args:
        img: torch.FloatTensor in shape of [N, C, H, W]
    """
    gray_factor = torch.tensor([0.114, 0.587, 0.299], dtype=torch.float32).view(1, 3, 1, 1)
    img_mean = torch.sum(img * gray_factor, dim=1, keepdim=True)
    gray_img = img_mean.repeat(1, 3, 1, 1).contiguous()
    return gray_img


def img_norm(img,
             mean=(0.485, 0.456, 0.406),
             std=(0.229, 0.224, 0.225),
             to_pil=True):
    """ Image normalization (pre-processing).

    Args:
        img (torch.Tensor): 4D image tensor (N, C, H, W), in OpenCV-format [BGR (0~255 uint8)]
        mean (iterable): normalized mean values.
        std (iterable): normalized std values.
        to_pil (bool): if convert to PIL format before normalization.

    Returns:
        img (torch.Tensor): normalized image tensor.

    """
    assert img.dim() == 4, "The input should be [N, C, H, W] format. Got ({})".format(img.size())
    if to_pil:
        img = img[:, [2, 1, 0], :, :].contiguous()
        img.div_(255.0)
    for i, (m, s) in enumerate(zip(mean, std)):
        img[:, i, :, :].sub_(m).div_(s)
    return img


def img_denorm(img,
               mean=(0.485, 0.456, 0.406),
               std=(0.229, 0.224, 0.225),
               to_pil=True):
    """ De-normalization

    Args:
        img (torch.Tensor): 4D normalized image tensor (N, C, H, W)
        mean (iterable): normalized mean values.
        std (iterable): normalized std values.
        to_pil (bool): if convert to PIL format before normalization.

    Returns:
        img (torch.Tensor): raw image tensor.

    """
    img = img.clone()
    for i, (m, s) in enumerate(zip(mean, std)):
        img[:, i, :, :].mul_(s).add_(m)
    if to_pil:
        img = img[:, [2, 1, 0], :, :].contiguous()
        img.mul_(255.0)
    return img


def img_pad(img, target_h, target_w):
    """ Pad image to the target shape

    Args:
        img (torch.Tensor): 4D normalized image tensor (N, C, H, W)
        target_h (int): output height
        target_w (int): output_width
    """

    img_h, img_w = img.size(2), img.size(3)
    assert img_h <= target_h and img_w <= target_w
    if img_h == target_h and img_w == target_w:
        return img
    sh = (target_h - img_h) // 2
    th = sh + img_h
    sw = (target_w - img_w) // 2
    tw = sw + img_w
    output = img.new_zeros(size=(img.size(0), img.size(1), target_h, target_w))
    output[:, :, sh:th, sw:tw] = img
    return output
