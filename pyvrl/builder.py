from mmcv import Registry, build_from_cfg

# Define different registry types to be used to map the class
DATASETS = Registry(name='dataset')
TRANSFORMS = Registry(name='transform')
BACKBONES = Registry(name='backbone')
MODELS = Registry(name='model')

def build_dataset(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, DATASETS, default_args= default_args)

def build_backbone(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, BACKBONES, default_args=default_args)


def build_transform(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, TRANSFORMS, default_args=default_args)

def build_head(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, HEAD, default_args=default_args)


def build_model(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, MODELS, default_args=default_args)




