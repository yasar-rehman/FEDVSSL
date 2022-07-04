import _init_paths

import cv2
import numpy as np
import os
import time
import torch
import argparse
# from mmcv import Config, init_dist, load_checkpoint,mkdir_or_exist
from mmcv import Config, mkdir_or_exist
from pyvrl.apis import get_root_logger, set_random_seed, train_network
from pyvrl.utils import img_tensor2np, visualization as uvis
from pyvrl.builder import build_model, build_dataset
from mmcv.runner import load_checkpoint
from collections import OrderedDict
from flwr.common import parameter

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

def parse_args():
    parser = argparse.ArgumentParser(description='Grad-CAM visualization')
    parser.add_argument('--cfg', default='configs/pot_v2/ft_ucf101/0717_r3d_ucf101_label.py', type=str, help='train config file path')
    parser.add_argument('--checkpoint', default=f'/home/root/yasar/cambridge_branch/federated-unsupervised-learning/videoSSL/reproduce_papers/action_recog_vcop/round_{540}/dtask_ucf/epoch_150.pth', 
    help='the checkpoint file to resume from')
    parser.add_argument('--data_dir', default='/home/data0/DATA/', type=str, help='the dir that save training data')
    args = parser.parse_args()

    return args


class FeatureExtractor(object):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs(object):
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            else:
                x = module(x)

        return target_activations, x


class GradCam(object):

    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.feature_module = feature_module

        self.model.eval()
        self.model.cuda()

        self.extractor = ModelOutputs(self.model.backbone, self.feature_module, target_layer_names)

    def forward(self, imgs):
        return self.model(imgs)

    def __call__(self, imgs, index=None):
        n, c, t, im_h, im_w = imgs.size()
        imgs = imgs.cuda()  # [N, C, T, H, W]

        features, output = self.extractor(imgs)
        output = output.mean(dim=[2, 3, 4], keepdim=True)
        output = self.model.cls_head(output)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(3, 4))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w.reshape(-1, 1, 1) * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam_resized = np.zeros((cam.shape[0], im_h, im_w), dtype=cam.dtype)
        for i in range(cam.shape[0]):
            cam_resized[i] = cv2.resize(cam[i], (im_w, im_h))
        cam_resized = cam_resized - np.min(cam_resized)
        cam_resized = cam_resized / np.max(cam_resized)
        return cam_resized


class GuidedBackpropReLU(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        feat = self.model.backbone(input)
        feat = feat.mean(dim=[2, 3, 4], keepdim=True)
        output = self.model.cls_head(feat)
        return output

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, ...]
        output = output.transpose((1, 2, 3, 0))
        return output


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    if 'pretrained' in cfg['model']['backbone']:
        cfg['model']['backbone']['pretrained'] = None
    if args.data_dir is not None:
        if 'test' in cfg.data:
            cfg.data.test.root_dir = args.data_dir
    
    assert os.path.exists(args.checkpoint)

    basename = os.path.basename(args.checkpoint)[:-4]
    output_dir = os.path.join(os.path.dirname(args.checkpoint), 'grad_cam_{}'.format(basename))
    mkdir_or_exist(output_dir)

    # build dataset
    dataset = build_dataset(cfg.data.test)

    print(cfg.model)

    # build model
    model1 = build_model(cfg.model, default_args=dict(train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg))
    model2 = build_model(cfg.model, default_args=dict(train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg))
    if args.checkpoint.endswith('.npz'):
        model1 = _load_checkpoint(model1, args.checkpoint)
        model2 = _load_checkpoint(model2, args.checkpoint)
    else:
        load_checkpoint(model1, args.checkpoint)
        load_checkpoint(model2, args.checkpoint)

    # wrapper by GradCam
    grad_cam = GradCam(model1, model1.backbone.layer4, ['0'])
    gb_model = GuidedBackpropReLUModel(model2, use_cuda=True)

    image_index_list = [i * len(dataset) // 101 for i in range(101)]
    # random_index_list = list(np.random.permutation(len(dataset)))
    for i in image_index_list[:50]:
        data = dataset[i]
        imgs = data['imgs'].data[0:1]
        cam = grad_cam(imgs)
        imgs.requires_grad_(True)
        gb = gb_model(imgs)
        imgs = img_tensor2np(imgs.detach()[0].permute(1, 0, 2, 3).contiguous())

        out_imgs = []
        for k in range(len(imgs)):
            i_img = imgs[k]
            i_gb = gb[k]
            i_cam = cam[k * len(cam) // len(imgs)]
            i_cam_gb = i_gb * np.expand_dims(i_cam, -1)
            cam_heatmap = cv2.applyColorMap(np.uint8(255 * i_cam), cv2.COLORMAP_JET)
            i_cam_gb = deprocess_image(i_cam_gb)
            i_gb = deprocess_image(i_gb)

            out_img = np.concatenate((i_img, cam_heatmap), axis=1)
            out_img = np.concatenate((out_img,
                                      np.concatenate((i_gb, i_cam_gb), axis=1)), axis=0)
            out_imgs.append(out_img)


        out_path = os.path.join(output_dir, '{:05d}.gif'.format(i))
        uvis.save_gif(out_imgs, out_path, 500)
        print(i)


if __name__ == '__main__':
    main()
