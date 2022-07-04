import numpy as np
import os
import torch
from torch.utils.data import Dataset
from typing import List, Union

# from cvlib import obj_from_dict, DataContainer
from mmcv.parallel import DataContainer
from mmcv.runner.utils import obj_from_dict

from ....datasets import storage_backends, transforms, data_catelog
from ....datasets.video_info import load_annotations
from ....builder import DATASETS


@DATASETS.register_module()
class TSNDataset_retrieval(Dataset):
    """ Sampling clips from video dataset.
    This code is mainly ported from mmaction lib.
    https://github.com/open-mmlab/mmaction

    Args:
        name (str): dataset name. it should be included in data_catelog.py
        root_dir (str): root directory that saves video datasets.
        backend (dict): storage backend configuration. It specifies how this
            dataset loads images, from raw frames? from zip file?
            or from video file. The type should be defined in '.storage_backends'
        transform_cfg (dict): image transformation configurations. each transform should be
            defined in '.transforms'
        modality (str or list[str]): output data modality. (support RGB only now)
        num_segments (int): how many sampling clips.
        sample_length (int): how many frames in each clip.
        sample_stride (int): frame-level temporal stride in each clip
        random_shift (int): if true and not in test mode, the starting frame of each clip
            will be randomly shifted.
        temporal_jitter (bool): when sample_stride > 2, apply for temporal
            jitter data augmentation.
        test_mode (bool): return training data or testing data.

    """
    def __init__(self,
                 name: str,
                 root_dir: str,
                 backend: dict,
                 transform_cfg: dict,
                 modality: Union[str, List[str]],
                 num_segments: int,
                 sample_length: int,
                 sample_stride: int,
                 random_shift: bool,
                 temporal_jitter: bool,
                 test_mode: bool):
        self.name = name
        self.root_path = data_catelog.get_data_dir(name, root_dir)
        self.video_infos = load_annotations(root_dir, self.name)
        # build storage backend
        self.backend = obj_from_dict(backend, storage_backends)
        self.img_transform = obj_from_dict(transform_cfg, transforms)

        # save modality
        if isinstance(modality, str):
            self.modalities = [modality]
        else:
            self.modalities = modality
        assert isinstance(self.modalities, (list, tuple))

        # frame sample settings
        self.num_segments = num_segments
        self.random_shift = random_shift
        self.sample_length = sample_length
        self.sample_stride = sample_stride
        self.base_length = sample_length * sample_stride
        self.temporal_jitter = temporal_jitter
        self.test_mode = test_mode

    def __getitem__(self, idx):
        """ Get sampling clips by given video index.
        This function will return a dict.
        In training mode (test_mode == False), the dict contains
        image data and ground-truth label data. While in test mode,
        only image data is returned.

        (Currently support RGB image data only)
        Image data is a torch Tensor in shape of [M, C, T, H, W], where
        M is [number of segments] * [number of over-sampling (by default 1)]

        Ground-truth label is a torch LongTensor in shape of [1, ], which
        indicated the class label of current sampled video.

        Both Image data and ground-truth data is wrapped by DataContainer object.
        This class is adopted from mmcv library, which is helpful to distributed training.

        """
        video_info = self.video_infos[idx]

        # build video storage backend object
        storage_obj = self.backend.open(video_info)  # type: storage_backends.BaseStorageBackend
        num_frames = len(storage_obj)

        # sample index from video
        frame_inds = self._sample_indices(num_frames)  # [num_segment, sample_length]

        # get frame according to the frame indexes.
        # each element in img_list is a numpy array in shape of [H, W, 3]
        img_list = storage_obj.get_frame(frame_inds.reshape(-1))

        # apply for image augmentation and transform to a torch Tensor
        img_tensor = self.img_transform.apply_image(img_list)  # type: torch.Tensor
        # (M, C, H, W) M = 1 * N_oversample * N_seg * L
        img_tensor = img_tensor.view((-1, self.num_segments, self.sample_length) + img_tensor.shape[1:])
        # N_over x N_seg x L x C x H x W
        img_tensor = img_tensor.permute(0, 1, 3, 2, 4, 5)
        # N_over x N_seg x C x L x H x W
        img_tensor = img_tensor.reshape((-1,) + img_tensor.shape[2:])
        # M' x C x L x H x W

        # construct data blob
        if len(self.modalities) > 1:
            # TODO(guangting): add supports for flow & rgb-diff
            raise NotImplementedError

        data = dict(
            imgs=DataContainer(img_tensor, stack=True, pad_dims=2, cpu_only=False),
        )

        if not self.test_mode:
            gt_label = torch.LongTensor([video_info['label']]) - 1
            data['gt_labels'] = DataContainer(gt_label, stack=True, pad_dims=None, cpu_only=False)

        return data

    def __len__(self):
        return len(self.video_infos)

    def _sample_indices(self, num_frames: int) -> np.ndarray:
        delta_length = num_frames - self.base_length + 1
        if self.random_shift and not self.test_mode:
            average_duration = delta_length // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration)
                offsets = offsets + np.random.randint(average_duration, size=self.num_segments)
            elif num_frames > max(self.num_segments, self.base_length):
                offsets = np.sort(np.random.randint(delta_length, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments, ), np.int)
        else:
            if delta_length > 0:
                tick = float(delta_length) / self.num_segments
                offsets = np.array([int(tick / 2.0 + tick * x)
                                    for x in range(self.num_segments)], np.int)
            else:
                offsets = np.zeros((self.num_segments, ), np.int)

        inds = np.arange(0, self.base_length, self.sample_stride, dtype=np.int).reshape(1, self.sample_length)
        if self.num_segments > 1:
            inds = np.tile(inds, (self.num_segments, 1))
        # apply for the init offset
        inds = inds + offsets.reshape(self.num_segments, 1)
        if self.temporal_jitter and self.sample_stride > 1:
            skip_offsets = np.random.randint(self.sample_stride, size=self.sample_length)
            inds = inds + skip_offsets.reshape(1, self.sample_length)
        inds = np.clip(inds, a_min=0, a_max=num_frames-1)
        inds = inds.astype(np.int)
        return inds
