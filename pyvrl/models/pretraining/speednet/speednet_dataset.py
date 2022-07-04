import random
import torch
from torch.utils.data import Dataset
from typing import List
from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder


@DATASETS.register_module()
class SpeedNetDataset(Dataset):
    """ use speed info to supervised the model training.

        This idea has been mentioned in many papers:
        [1] SpeedNet: Learning the Speediness in Videos, CVPR'20
            https://arxiv.org/abs/2004.06130
        [2] Self-Supervised Spatio-Temporal Representation Learning Using Variable
            Playback Speed Prediction, arXiv 20'03
            https://arxiv.org/abs/2003.02692
        [3] Video Playback Rate Perception for Self-supervisedSpatio-Temporal
            Representation Learning, CVPR'20
            https://arxiv.org/abs/2006.11476
    """

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 clip_len: int,
                 clip_strides: List[int],
                 transform_cfg: list,
                 test_mode: bool = False,
                 name: str=None):
        if name is None:
            name = 'undefined_dataset'
        self.name = name
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source,
                                                     dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.img_transform = Compose(transform_cfg)
        self.test_mode = test_mode
        self.clip_len = clip_len
        self.clip_strides = clip_strides

    def _sample_frame_inds(self, num_frames):
        stride_idx = random.choice(list(range(len(self.clip_strides))))
        stride = self.clip_strides[stride_idx]
        pos_stride = stride if stride >= 0 else -stride

        total_len = pos_stride * (self.clip_len - 1) + 1
        if total_len >= num_frames:
            start_index = 0
        else:
            start_index = random.randint(0, num_frames - total_len)
        frame_inds = [min(start_index + i * pos_stride, num_frames - 1)
                      for i in range(self.clip_len)]

        if stride < 0:
            frame_inds = list(reversed(frame_inds))

        return frame_inds, stride_idx

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        # build video storage backend object
        storage_obj = self.backend.open(video_info)

        imgs, stride_idx = self.sample_single_img_tensor(storage_obj)

        imgs = imgs.unsqueeze(0)
        gt_label = torch.LongTensor([stride_idx])
        data = dict(
            imgs=DataContainer(imgs, stack=True, pad_dims=1, cpu_only=False),
            gt_labels=DataContainer(gt_label,
                                    stack=True, pad_dims=None, cpu_only=False)
        )
        storage_obj.close()

        return data

    def sample_single_img_tensor(self, storage_obj):
        frame_inds, stride_idx = self._sample_frame_inds(len(storage_obj))
        frames = storage_obj.get_frame(frame_inds)
        # apply for transforms
        frames, trans_params = \
            self.img_transform.apply_image(frames, return_transform_param=True)
        img_tensor = frames.permute(1, 0, 2, 3).contiguous()
        return img_tensor, stride_idx

    def evaluate(self, results, logger=None):
        if isinstance(results, list):
            if results[0].ndim == 1:
                results = [r[np.newaxis, ...] for r in results]
            results = np.concatenate(results, axis=0)
        assert len(results) == len(self), \
            f'The results should have same size as gts. But' \
            f' got {len(results)} and {len(self)}'
        labels = np.array([int(self.data_source[_]['label']) - 1
                           for _ in range(len(self))], np.long)
        sort_inds = results.argsort(axis=1)[:, ::-1]

        acc_dict = dict()
        for k in [1, 5]:
            top_k_inds = sort_inds[:, :k]
            correct = (top_k_inds.astype(np.long) ==
                       labels.reshape(len(self), 1))
            correct_count = np.any(correct, axis=1).astype(np.float32).sum()
            acc = correct_count / len(self)
            acc_dict[f'top_{k}_acc'] = acc
            if logger is not None:
                logger.info(f'top_{k}_acc: {acc*100}%')

        # mean class accuracy
        per_class_acc = dict()
        for i in range(len(results)):
            class_id = int(labels[i])
            if class_id not in per_class_acc:
                per_class_acc[class_id] = []
            if sort_inds[i, 0] == class_id:
                per_class_acc[class_id].append(1.0)
            else:
                per_class_acc[class_id].append(0.0)
        per_class_acc_list = []
        for k, v in per_class_acc.items():
            per_class_acc_list.append(sum(v) / len(v))
        acc_dict[f'mean_class_acc'] = sum(per_class_acc_list) / \
                                      len(per_class_acc_list)

        return acc_dict
