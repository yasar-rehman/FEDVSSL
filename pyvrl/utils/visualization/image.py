import cv2
import numpy as np
import os
from typing import Tuple, List, Union
from PIL import Image


def draw_boxes(img: np.ndarray,
               boxes: np.ndarray,
               colors: Union[Tuple, List[Tuple]] = None):
    assert img.ndim == 3 and img.shape[2] == 3
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
    num = boxes.shape[0]
    if colors is None:
        colors = [(0, 255, 0) for _ in range(num)]
    elif isinstance(colors, tuple):
        colors = [colors for _ in range(num)]

    for i in range(num):
        box = np.round(boxes[i]).astype(int)
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=colors[i], thickness=2)
    if isinstance(img, cv2.UMat):
        img = img.get()
    return img


def group_img_list(img_list: List[np.ndarray]):
    img_height, img_width = img_list[0].shape[0:2]
    num = len(img_list)
    ncol = int(np.ceil(np.sqrt(num)))
    nrow = int(np.ceil(num / ncol))
    img = np.zeros((nrow * img_height, ncol * img_width, 3), np.uint8)

    for i in range(num):
        c, r = i % ncol, i // ncol
        img[r*img_height:(r+1)*img_height, c*img_width:(c+1)*img_width, :] = img_list[i]
    return img


def save_gif(image_list, output_path, duration=120):
    out_imgs = [Image.fromarray(img[:, :, ::-1].astype(np.uint8)) for img in image_list]
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    out_imgs[0].save(output_path, save_all=True, append_images=out_imgs[1:], duration=120, loop=0)
