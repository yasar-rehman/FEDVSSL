
import _init_paths
import os
from addict import Dict
import argparse

# from action.builder import build_model, build_dataset
from pyvrl.builder import build_model, build_dataset
from pyvrl.apis import get_root_logger, test_clip_retrieval
from collections import OrderedDict
# from action.apis import test_clip_retrieval, get_root_logger
# from cvlib import Config
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation a action recognizer')
    parser.add_argument('--cfg', 
    default='', 
    type=str, help='train config file path')
    parser.add_argument('--dataset_name', default='ucf101', type=str, help='dataset type')
    parser.add_argument('--work_dir',default='', 
    help='the dir to save logs and models')
    parser.add_argument('--data_dir', default='/home/data3/DATA', type=str, help='the dir that save training data')
    parser.add_argument('--checkpoint',
    default='',
    type=str, help='the checkpoint file to resume from')
    parser.add_argument('--gpus', type=int, default=1,
                        help='number of gpus to use '
                             '(only applicable to non-distributed training)')
    parser.add_argument('--batchsize', type=int, default=6)
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    return args


def prepare_model_config(backbone):
    print("preparing model")
    model = dict(
        type='RetrievalTSN3D',
        compress=dict(
            type='AdaptiveAvgPool3d',
            output_size=[2, 3, 3],
        )
    )
    model['backbone'] = backbone
    print("model prepared")
    return Dict(model)


def prepare_data_config(dataset_name, root_dir):

    if dataset_name == 'ucf101':
        train_dataset_name = 'ucf101_train_split1'
        test_dataset_name = 'ucf101_test_split1'
    elif dataset_name == 'hmdb51':
        train_dataset_name = 'hmdb51_train_split1'
        test_dataset_name = 'hmdb51_test_split1'
    else:
        raise NotImplementedError

    data = dict(
        videos_per_gpu=8,
        workers_per_gpu=4,
        train=dict(
            type='TSNDataset_retrieval',
            name=train_dataset_name,
            root_dir=root_dir,
            backend=dict(type='ZipBackend'),
            modality='RGB',
            num_segments=10,
            sample_length=16,
            sample_stride=2,
            random_shift=False,
            temporal_jitter=False,
            test_mode=True,
            transform_cfg=dict(
                type='Compose',
                transform_cfgs=[
                    dict(type='GroupScale', scales=[(171, 128)]),
                    dict(type='GroupCenterCrop', out_size=112),
                    dict(
                        type='GroupToTensor',
                        switch_rgb_channels=True,
                        div255=True,
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    )
                ]
            )
        ),
        test=dict(
            type='TSNDataset_retrieval',
            name=test_dataset_name,
            root_dir=root_dir,
            backend=dict(type='ZipBackend'),
            modality='RGB',
            num_segments=10,
            sample_length=16,
            sample_stride=2,
            random_shift=False,
            temporal_jitter=False,
            test_mode=True,
            transform_cfg=dict(
                type='Compose',
                transform_cfgs=[
                    dict(type='GroupScale', scales=[(171, 128)]),
                    dict(type='GroupCenterCrop', out_size=112),
                    dict(
                        type='GroupToTensor',
                        switch_rgb_channels=True,
                        div255=True,
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    )
                ]
            )
        ),
    )
    return Dict(data)


if __name__ == '__main__':
    logger = get_root_logger(log_level='INFO')

    args = parse_args()
    _cfg = Config.fromfile(args.cfg)
    # if args.data_dir is not None:
    #     if 'train' in _cfg.data:
    #         _cfg.data.train.data_dir = args.data_dir
    #     if 'val' in _cfg.data:
    #         _cfg.data.val.data_dir = args.data_dir
    #     if 'test' in _cfg.data:
    #         _cfg.data.test.data_dir = args.data_dir
    
    cfg = dict(
        model=prepare_model_config(_cfg['model']['backbone']),
        data=prepare_data_config(args.dataset_name, args.data_dir)
        # data = _cfg.data
    )
    cfg = Dict(cfg)
    cfg.gpus = args.gpus
    cfg.data.videos_per_gpu = args.batchsize

    cfg.work_dir = args.work_dir

    

    if args.checkpoint is None:
        load_from = cfg.work_dir
    else:
        load_from = args.checkpoint
    print(load_from)
    

    if load_from:
        chkpt_list = [load_from]
    # elif os.path.isdir(load_from):
    #     chkpt_list = [os.path.join(load_from, fn)
    #                   for fn in os.listdir(load_from) if fn.endswith('.npz') and not fn.startswith('latest')]
    # eli)f os.path.isfile(load_from:
    #     chkpt_list = [load_from]
    else:
        raise FileNotFoundError("Cannot find {}".format(args.checkpoint))

    print(chkpt_list)
    cfg.work_dir = os.path.join(cfg.work_dir, 'clip_retrieval')

    # build a dataloader
    model = build_model(cfg.model, default_args=dict(train_cfg=None, test_cfg=None))
    train_dataset = build_dataset(cfg.data.train)
    test_dataset = build_dataset(cfg.data.test)
    

    results = test_clip_retrieval(model,
                                  train_dataset,
                                  test_dataset,
                                  cfg,
                                  chkpt_list,
                                  logger=logger,
                                  progress=args.progress)
