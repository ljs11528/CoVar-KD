"""Model store which handles pretrained models """

from .pspnet import *
from .deeplabv3 import *
from .deeplabv3_mobile import *
from .psp_mobile import *
from .deeplabv3_mobilenetv3 import *
from .upernet import *

__all__ = ['get_segmentation_model']


def get_segmentation_model(model, **kwargs):
    if model == 'segformer':
        # Lazy import to avoid pulling timm/torchvision/triton unless needed
        from .segformer import get_segformer
        return get_segformer(**kwargs)

    models = {
        'psp': get_psp,
        'deeplabv3': get_deeplabv3,
        'deeplab_mobile': get_deeplabv3_mobile,
        'psp_mobile': get_psp_mobile,
        'deeplabv3_mobilenet_ssseg': get_deeplabv3_mobilenet_ssseg,
        'upernet': get_upernet,
        'upernet_lite': get_upernet_lite,
    }
    return models[model](**kwargs)
