""" Dataset configurations. """
import os

_DATASETS = dict(
    ucf101_mini=dict(
        root='ucf101',
        ann='ucf101/annotations/train_mini.txt',
        frame_zip_path='ucf101/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='ucf101/zips/{}_tvl1.zip',
        flow_x_fmt='{:05d}_x.png',
        flow_y_fmt='{:05d}_y.png'
    ),
    ucf101_train_split1=dict(
        root='ucf101',
        ann='ucf101/annotations/train_split_1.txt',
        frame_zip_path='ucf101/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='ucf101/zips/{}_tvl1.zip',
        flow_x_fmt='{:05d}_x.png',
        flow_y_fmt='{:05d}_y.png'
    ),
    ucf101_train_split1_percent10=dict(
        root='ucf101',
        ann='ucf101/annotations/train_split_1_percent10.txt',
        frame_zip_path='ucf101/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='ucf101/zips/{}_tvl1.zip',
        flow_x_fmt='{:05d}_x.png',
        flow_y_fmt='{:05d}_y.png'
    ),
    ucf101_train_split1_percent25=dict(
        root='ucf101',
        ann='ucf101/annotations/train_split_1_percent25.txt',
        frame_zip_path='ucf101/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='ucf101/zips/{}_tvl1.zip',
        flow_x_fmt='{:05d}_x.png',
        flow_y_fmt='{:05d}_y.png'
    ),
    ucf101_train_split1_percent50=dict(
        root='ucf101',
        ann='ucf101/annotations/train_split_1_percent50.txt',
        frame_zip_path='ucf101/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='ucf101/zips/{}_tvl1.zip',
        flow_x_fmt='{:05d}_x.png',
        flow_y_fmt='{:05d}_y.png'
    ),
    ucf101_train_split1_percent75=dict(
        root='ucf101',
        ann='ucf101/annotations/train_split_1_percent75.txt',
        frame_zip_path='ucf101/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='ucf101/zips/{}_tvl1.zip',
        flow_x_fmt='{:05d}_x.png',
        flow_y_fmt='{:05d}_y.png'
    ),
    ucf101_test_split1=dict(
        root='ucf101',
        ann='ucf101/annotations/test_split_1.txt',
        frame_zip_path='ucf101/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='ucf101/zips/{}_tvl1.zip',
        flow_x_fmt='{:05d}_x.png',
        flow_y_fmt='{:05d}_y.png'
    ),
    hmdb51_train_split1=dict(
        root='hmdb51',
        ann='hmdb51/annotations/train_split_1.txt',
        frame_zip_path='hmdb51/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
    ),
    hmdb51_test_split1=dict(
        root='hmdb51',
        ann='hmdb51/annotations/test_split_1.txt',
        frame_zip_path='hmdb51/zips/{}.zip',
        frame_fmt='img_{:05d}.jpg',
    ),
    kinetics400_train=dict(
        root='kinetics400',
        ann='kinetics400/train_videofolder.txt',
        frame_zip_path='kinetics400/{}/RGB_frames.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='kinetics400_flow/{}/Flow_frames.zip',
        flow_x_fmt='x_{:05d}.jpg',
        flow_y_fmt='y_{:05d}.jpg'
    ),
    kinetics400_train_10=dict(
        root='kinetics400',
        ann='kinetics400/train_videofolder_10.txt',
        frame_zip_path='kinetics400/{}/RGB_frames.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='kinetics400_flow/{}/Flow_frames.zip',
        flow_x_fmt='x_{:05d}.jpg',
        flow_y_fmt='y_{:05d}.jpg'
    ),
    kinetics400_train_25=dict(
        root='kinetics400',
        ann='kinetics400/train_videofolder_25.txt',
        frame_zip_path='kinetics400/{}/RGB_frames.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='kinetics400_flow/{}/Flow_frames.zip',
        flow_x_fmt='x_{:05d}.jpg',
        flow_y_fmt='y_{:05d}.jpg'
    ),
    kinetics400_train_50=dict(
        root='kinetics400',
        ann='kinetics400/train_videofolder_50.txt',
        frame_zip_path='kinetics400/{}/RGB_frames.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='kinetics400_flow/{}/Flow_frames.zip',
        flow_x_fmt='x_{:05d}.jpg',
        flow_y_fmt='y_{:05d}.jpg'
    ),
    kinetics400_train_75=dict(
        root='kinetics400',
        ann='kinetics400/train_videofolder_75.txt',
        frame_zip_path='kinetics400/{}/RGB_frames.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='kinetics400_flow/{}/Flow_frames.zip',
        flow_x_fmt='x_{:05d}.jpg',
        flow_y_fmt='y_{:05d}.jpg'
    ),
    kinetics400_val=dict(
        root='kinetics400',
        ann='kinetics400/val_videofolder.txt',
        frame_zip_path='kinetics400/{}/RGB_frames.zip',
        frame_fmt='img_{:05d}.jpg',
        flow_zip_path='kinetics400_flow/{}/Flow_frames.zip',
        flow_x_fmt='x_{:05d}.jpg',
        flow_y_fmt='y_{:05d}.jpg'
    ),
    something_v1_train=dict(
        root='something-something-v1',
        ann='something-something-v1/train_videofolder.txt',
        frame_zip_path='something-something-v1/{}/RGB_frames.zip',
        frame_fmt='{:05d}.jpg',
    ),
    something_v1_val=dict(
        root='something-something-v1',
        ann='something-something-v1/val_videofolder.txt',
        frame_zip_path='something-something-v1/{}/RGB_frames.zip',
        frame_fmt='{:05d}.jpg',
    ),
)


def get_ann_fn(name: str, root_dir: str = None):
    """ Retrieve the annotation file for the dataset."""
    if root_dir is not None:
        return os.path.join(root_dir, _DATASETS[name]['ann'])
    else:
        return _DATASETS[name]['root']


def get_data_dir(name: str, root_dir: str = None):
    """ Retrieve the data root directory path. """
    if root_dir is not None:
        return os.path.join(root_dir, _DATASETS[name]['root'])
    else:
        return _DATASETS[name]['root']


def get_frame_zip_path(dataset_name: str, video_name: str, root_dir: str):
    return os.path.join(root_dir,
                        _DATASETS[dataset_name]['frame_zip_path'].format(video_name))


def get_flow_zip_path(dataset_name: str, video_name: str, root_dir: str):
    dataset = _DATASETS[dataset_name]
    if 'flow_zip_path' in dataset:
        return os.path.join(root_dir, dataset['flow_zip_path'].format(video_name))
    else:
        return None


def get_flow_x_fmt(dataset_name: str):
    return _DATASETS[dataset_name].get('flow_x_fmt', None)


def get_flow_y_fmt(dataset_name: str):
    return _DATASETS[dataset_name].get('flow_y_fmt', None)


def get_frame_fmt(dataset_name: str):
    return _DATASETS[dataset_name]['frame_fmt']
