import numpy as np
from typing import Iterable, List


class BaseStorageItem(object):

    def get_frame(self, indices: Iterable[int]) -> List[np.ndarray]:
        raise NotImplementedError

    def get_flow(self, indices: Iterable[int]) -> List[np.ndarray]:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class BaseStorageBackend(object):

    def open(self, video_info) -> BaseStorageItem:
        raise NotImplementedError
