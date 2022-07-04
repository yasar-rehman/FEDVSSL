import torch
import numpy as np

from typing import Union


def center_crop(t: Union[torch.Tensor, np.ndarray],
                crop_height: int,
                crop_width: int = None):
    """ Given the output size (crop_height, crop_width), center crop the target tensor
    """
    if crop_width is None:
        crop_width = crop_height

    if isinstance(t, torch.Tensor):
        return center_crop_torch(t, crop_height, crop_width)
    elif isinstance(t, np.ndarray):
        return center_crop_np(t, crop_height, crop_width)
    else:
        raise NotImplementedError


def center_crop_np(t: np.ndarray,
                   crop_height: int,
                   crop_width: int):
    h, w, c = t.shape
    if h == crop_height and w == crop_width:
        return t
    sy = (h - crop_height) // 2
    ty = sy + crop_height
    sx = (w - crop_width) // 2
    tx = sx + crop_width
    crop_tensor = np.ascontiguousarray(t[sy:ty, sx:tx, ...])
    return crop_tensor


def center_crop_torch(t: torch.Tensor,
                      crop_height: int,
                      crop_width: int):
    n, c, h, w = t.size()
    if h == crop_height and w == crop_width:
        return t
    sy = (h - crop_height) // 2
    ty = sy + crop_height
    sx = (w - crop_width) // 2
    tx = sx + crop_width
    crop_tensor = t[:, :, sy:ty, sx:tx].contiguous()
    return crop_tensor


