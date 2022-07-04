import os
import pickle
import cv2
import numpy as np
import zipfile
from typing import List
from .base_storage_backend import BaseStorageBackend
from .base_storage_backend import BaseStorageItem

# import third_party.optical_flow as flow_utils


class ZipItem(BaseStorageItem):
    """ Zip storage item for loading images from a video.
    Each video clip has one corresponding zip file, which stores
    the video frames (like 00001.jpg, 00002.jpg, ...) or flow information.
    """

    def __init__(self, video_info: dict):
        # Get zip path from the video information
        self.frame_zip_path = video_info.get('frame_zip', None)
        self.frame_fmt = video_info.get('frame_fmt', 'img_{:05d}.jpg')

        self.flow_zip_path = video_info.get('flow_zip', None)
        self.flow_x_fmt = video_info.get('flow_x_fmt', 'x_{:05d}.jpg')
        self.flow_y_fmt = video_info.get('flow_y_fmt', 'y_{:05d}.jpg')

        # zipfile descriptor (during initialization, it's None)
        self.frame_zip_fid = None
        self.flow_zip_fid = None

    def __len__(self):
        if self.frame_zip_fid is None:
            self._check_available(self.frame_zip_path)
            self.frame_zip_fid = zipfile.ZipFile(self.frame_zip_path, 'r')
        namelist = self.frame_zip_fid.namelist()
        namelist = [name for name in namelist if name.endswith('.jpg')]
        return len(namelist)

    def close(self):
        if self.frame_zip_fid is not None:
            self.frame_zip_fid.close()
        if self.flow_zip_fid is not None:
            self.flow_zip_fid.close()

    def get_frame(self, indices: List[int]) -> List[np.ndarray]:
        """ Load image frames from the given zip file.
        Args:
            indices: frame index list (0-based index)
        Returns:
            img_list: the loaded image list, each element is a np.ndarray in shape of [H, W 3]
        """
        if isinstance(indices, int):
            indices = [indices]
        img_list = []
        if self.frame_zip_fid is None:
            self._check_available(self.frame_zip_path)
            self.frame_zip_fid = zipfile.ZipFile(self.frame_zip_path, 'r')

        for idx in indices:
            file_name = self.frame_fmt.format(int(idx) + 1)
            img = self.load_image_from_zip(self.frame_zip_fid, file_name, cv2.IMREAD_COLOR)
            img_list.append(img)
        return img_list

    def get_flow(self, indices: List[int]) -> List[np.ndarray]:
        """ Load flow from the zip file.
        Args:
            indices: frame index list
        Returns:
            flow_list: the loaded optical flow, each element is a np array in shape of [H, W, 2]
        """

        if isinstance(indices, int):
            indices = [indices]
        flow_list = []

        for i in range(len(indices)-1):
            cur_index = indices[i]
            next_index = indices[i+1]

            assert cur_index <= next_index

            if cur_index == next_index:
                first_flow = self.load_flow_from_zip(0)
                flow = np.zeros_like(first_flow)
            else:
                flow = None

            while cur_index < next_index:
                cur_flow = self.load_flow_from_zip(cur_index)
                if flow is None:
                    flow = cur_flow
                else:
                    flow = flow_utils.flow_propagation(flow, cur_flow)
                cur_index += 1

            flow_list.append(flow)
        return flow_list

    def load_flow_from_zip(self, index):
        """ Loading pre-processed optical flow from Zip files. """
        if self.flow_zip_fid is None:
            self._check_available(self.flow_zip_path)
            self.flow_zip_fid = zipfile.ZipFile(self.flow_zip_path, 'r')

        # the flow file name is 1-index based while the input index is 0-index based
        flow_x_name = self.flow_x_fmt.format(index + 1)
        flow_y_name = self.flow_y_fmt.format(index + 1)
        flow_x = self.load_image_from_zip(self.flow_zip_fid, flow_x_name, cv2.IMREAD_GRAYSCALE)
        flow_y = self.load_image_from_zip(self.flow_zip_fid, flow_y_name, cv2.IMREAD_GRAYSCALE)

        # by default, the flow is linearly scaled from [-20, 20] to [0, 255] and saved as image
        # we need to re-transform to the real scale range.

        flow_x = flow_x.astype(np.float32) * (40.0 / 255.0) - 20.0
        flow_y = flow_y.astype(np.float32) * (40.0 / 255.0) - 20.0

        # concat the flow x, y
        flow = np.concatenate((np.expand_dims(flow_x, axis=-1),
                               np.expand_dims(flow_y, axis=-1)),
                              axis=-1)
        return flow

    @staticmethod
    def load_image_from_zip(zip_fid, file_name, flag=cv2.IMREAD_COLOR):
        file_content = zip_fid.read(file_name)
        img = cv2.imdecode(np.fromstring(file_content, dtype=np.uint8), flag)
        return img

    @staticmethod
    def _check_available(zip_path):
        if zip_path is None:
            raise ValueError("There is not file path defined in video annotations")
        if not os.path.isfile(zip_path):
            raise FileNotFoundError("Cannot find zip file {}".format(zip_path))


class ZipBackend(BaseStorageBackend):

    def open(self, video_info) -> BaseStorageItem:
        return ZipItem(video_info)
