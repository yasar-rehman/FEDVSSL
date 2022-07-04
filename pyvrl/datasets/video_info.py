from typing import List
from . import data_catelog


def load_annotations(data_root: str,
                     dataset_name: str) -> List[dict]:
    """ Load some necessary annotations for training & testing.
    Args:
        data_root (str): data root directory path.
        dataset_name (str): dataset name
    """
    is_kinetics = 'kinetics' in dataset_name
    is_something = 'something' in dataset_name

    ann_file_path = data_catelog.get_ann_fn(dataset_name, data_root)
    video_info_list = []
    if ann_file_path.endswith('.txt'):
        with open(ann_file_path, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            if line.strip() == '':
                continue
            split = line.split(' ')
            if is_kinetics or is_something:
                name = ' '.join(split[:-2])
                nframe = int(split[-2])
                label = int(split[-1]) + 1  # this is not a bug because the label range is [0, 399]
                if nframe <= 3:
                    continue
            else:
                assert len(split) == 2
                name, label = split[0], int(split[1])
            video_info = dict(
                name=name,
                label=label,
                frame_zip=data_catelog.get_frame_zip_path(dataset_name, name, data_root),
                frame_fmt=data_catelog.get_frame_fmt(dataset_name),
                flow_zip=data_catelog.get_flow_zip_path(dataset_name, name, data_root),
                flow_x_fmt=data_catelog.get_flow_x_fmt(dataset_name),
                flow_y_fmt=data_catelog.get_flow_y_fmt(dataset_name),
            )
            video_info_list.append(video_info)
    else:
        raise NotImplementedError
    return video_info_list
