# Copyright 2018-2019 Open-MMLab. All rights reserved.

import torch
import os
import mmcv
from mmcv import mkdir_or_exist, fileio
import time
import re
from mmcv.runner import load_checkpoint
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel, collate
from .env import get_root_logger
from .test import multi_gpu_test, single_gpu_test
from ..datasets.dataloader import build_dataloader
from mmcv.utils import ProgressBar
import numpy as np
import logging
from math import ceil
from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader, Dataset
from .env import set_random_seed
from ..core.evaluation.accuracy import top_k_accuracy
from flwr.common import parameter
from collections import OrderedDict

def _load_checkpoint(_model, _chk_path):
    params = np.load(_chk_path, allow_pickle=True)
    params = params['arr_0'].item()
    params = parameter.parameters_to_weights(params)
    # _load_checkpoint(_model, params)
    print(len(params), len(_model.state_dict().keys()))
    params_dict = zip(_model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    print(f"length of the ordered dict_keys is: {len(state_dict.keys())}")
    _model.load_state_dict(state_dict, strict=True)

    return _model

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)

class _DataWrapper(Dataset):

    def __init__(self, dataset, batchsize: int = 1):
        self.dataset = dataset
        self.batchsize = batchsize
        self.num = int(ceil(len(self.dataset) / float(self.batchsize))) * self.batchsize

    def __getitem__(self, item):
        idx = item % (len(self.dataset))
        return self.dataset[idx]

    def __len__(self):
        return self.num



def test_network(model,
                 dataset,
                 cfg,
                 args,
                 distributed=False,
                 logger=None,
                 progress=True):
    
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        chk_name_list = [fn for fn in os.listdir(cfg.work_dir) if fn.endswith('.pth')]
        chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('epoch')]
        chk_epoch_list.sort()
        checkpoint = os.path.join(cfg.work_dir, f'epoch_{chk_epoch_list[-1]}.pth')
    cfg.checkpoint = checkpoint
    
    if logger is None:
        logger = get_root_logger(cfg.log_level)
    logger.info(f"Ckpt path: {cfg.checkpoint}")
    if cfg.checkpoint.endswith('.pth'):
        prefix = os.path.basename(cfg.checkpoint)[:-4]
    elif cfg.checkpoint.endswith('.npz'):
        prefix = os.path.basename(cfg.checkpoint)[:-4]
    else:
        prefix = 'unspecified'

    out_name = f'eval_{dataset.__class__.__name__}_{dataset.name}'
    output_dir = os.path.join(cfg.work_dir, out_name)
    mmcv.mkdir_or_exist(output_dir)

    cache_path = os.path.join(output_dir, f'{prefix}_results.pkl')
    if os.path.isfile(cache_path):
        logger.info(f"Load results from {cache_path}")
        results = mmcv.load(cache_path)
    else:
        if cfg.checkpoint.endswith('.npz'):
            model = _load_checkpoint(model, cfg.checkpoint)
            logger.info("The model is being loaded...")
            # _load_checkpoint(model, cfg.checkpoint, logger=logger)
        else:
            load_checkpoint(model, cfg.checkpoint, logger=logger)
        # save config in model
        model.cfg = cfg
        # build dataloader
        multiprocessing_context = None
        if cfg.get('numpy_seed_hook', True) and cfg.data.workers_per_gpu > 0:
            multiprocessing_context = 'spawn'
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            multiprocessing_context=None
        )

        # start inference
        if distributed:
            if cfg.get('syncbn', False):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False
            ) 
            results = multi_gpu_test(model, data_loader, progress=progress)
        
        else:
            model = MMDataParallel(model, device_ids=[0])
            
            results = single_gpu_test(model, data_loader, progress=progress)
            # for the down-stream task uncomment the below lines
            # eval_results = dataset.evaluate(results, logger)
            # results = eval_results

        
    
        
    
    # for the down-stream task comment this line
    results = sum(results)/len(results)
        # result_tosave = results.copy()
        # mmcv.dump(results, cache_path)
    print(" \n The result is:", results)
    mmcv.dump(results, os.path.join(output_dir,
                                            f'{prefix}_eval_results.json'))

    # evaluate results
    # eval_results = dataset.evaluate(results, logger)
    # rank, world_size = get_dist_info()
    # if rank == 0:
    #     # if results != None:
    #     #     results = sum(results)/len(results)
    #     print('\n',results)
        
    #     time.sleep(2)  # This line can prevent deadlock problem in some cases.
       
    return results
        
@torch.no_grad()
def test_clip_retrieval_perturb(model,
                        train_dataset,
                        test_dataset,
                        cfg,
                        args,
                        checkpoint,
                        logger=None,
                        progress=False):
    if logger is None:
        logger = logging.getLogger()

    def make_dataloader(_dataset, _cfg):
        return DataLoader(
            dataset=_DataWrapper(_dataset, batchsize=_cfg.gpus * _cfg.data.videos_per_gpu),
            batch_size=_cfg.gpus * _cfg.data.videos_per_gpu,
            shuffle=False,
            num_workers=_cfg.gpus * _cfg.data.workers_per_gpu,
            collate_fn=partial(collate, samples_per_gpu=_cfg.data.videos_per_gpu),
            pin_memory=True,
            drop_last=False
        )

    def extrac_features(_model, _dataloader, _progress=False):
        if _progress:
            progress_bar = ProgressBar(len(_dataloader))
        feature_pools = None
        for idx, data_batch in enumerate(_dataloader):
            i_features = _model(**data_batch)
            i_features = i_features.cpu().detach()
            if feature_pools is None:
                b, n, c = i_features.size()
                feature_pools = torch.zeros((b * len(_dataloader), n, c)).float()
            feature_pools[idx * b:(idx+1) * b, ...] = i_features
            if _progress:
                progress_bar.update()

        return feature_pools

    def process_single_chkpt(_model, _chk_path, _train_dataset, _test_dataset, _cfg, _progress=False):
        logger.info("Load checkpoint from {}".format(_chk_path))
        if _chk_path is not None:
            if _chk_path.endswith('.npz'):
                _model = _load_checkpoint(_model, _chk_path)
                logger.info("The model is being loaded...")
            else:
                logger.info("The model is being loaded...")
                load_checkpoint(_model, _chk_path)
        
        with torch.no_grad():
            logger.info("Adding perturbation")
            for param in _model.parameters():
                torch.manual_seed(0)
                param.add_((torch.randn(param.size())*args.ran_multip).cuda())



        logger.info("Extract training feature embeddings...")
        train_dataloader = make_dataloader(_train_dataset, _cfg)
        train_features = extrac_features(_model, train_dataloader, _progress)
        train_features = train_features[:len(_train_dataset)]

        logger.info("Extract test feature embeddings...")
        test_dataloader = make_dataloader(_test_dataset, cfg)
        test_features = extrac_features(_model, test_dataloader, _progress)
        test_features = test_features[:len(_test_dataset)]

        logger.info("Calculate pair-wise distance...")
        train_features = train_features.mean(dim=1).cuda()
        test_features = test_features.mean(dim=1).cuda()

        inner_prod = torch.matmul(test_features, train_features.transpose(0, 1))
        train_norm = train_features.norm(p=None, dim=1)
        cos_dist = inner_prod / train_norm.view(1, -1)  # [N_test, N_train]

        cos_dist = cos_dist.cpu().numpy()
        indices = np.argsort(cos_dist, axis=1)[:, ::-1]

        train_labels = np.array([v['label'] for v in _train_dataset.video_infos], dtype=int)
        test_labels = np.array([v['label'] for v in _test_dataset.video_infos], dtype=int)
        assign_labels = train_labels[indices]

        results = dict(
            train_labels=train_labels,
            test_labels=test_labels,
            assign_labels=assign_labels
        )

        for nk in [1, 5, 10, 20, 50]:
            correct = test_labels.reshape(-1, 1) == assign_labels[:, 0:nk]
            correct = np.any(correct, axis=1)
            num_correct = correct.astype(np.float32).sum()
            accuracy = num_correct / test_labels.shape[0]
            results['top{}_acc'.format(nk)] = accuracy

        return results

    # Step 1, setup model and checkpoints
    set_random_seed(0)
    if isinstance(checkpoint, str):
        checkpoint = [checkpoint]
    if not isinstance(model, MMDataParallel):
        model = MMDataParallel(model)
    model.cuda()
    model.eval()

    for chk_path in checkpoint:
        base_name = os.path.basename(chk_path)[:-4] if chk_path is not None else 'none'
        res_path = os.path.join(cfg.work_dir,
                                test_dataset.name,
                                'clip_retrieval_{}.pkl'.format(base_name))
      
        if os.path.exists(res_path):
            logger.info("Load cached results {}".format(res_path))
            dataset_results = fileio.load(res_path)
        else:
            # train_labels = np.array([v['label'] for v in train_dataset.video_infos], dtype=int)
            dataset_results = process_single_chkpt(model, chk_path, train_dataset, test_dataset, cfg, progress)
            mkdir_or_exist(os.path.dirname(res_path))
            fileio.dump(dataset_results, res_path)
        txt_path = os.path.join(cfg.work_dir,
                                test_dataset.name,
                                'clip_retrieval_acc_{}.txt'.format(base_name))
        with open(txt_path, 'w') as f:
            for k in dataset_results.keys():
                if k.startswith('top'):
                    f.write('{}: {}\n'.format(k, dataset_results[k]))

        logger.info('-----------------------')
        logger.info('chk: {}'.format(chk_path))
        for k in dataset_results.keys():
            if k.startswith('top'):
                logger.info('{}: {}'.format(k, dataset_results[k]))