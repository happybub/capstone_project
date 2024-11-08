import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import init


def get_config(parser=None):
    import config

    # first, update the configurations in config.py
    config_map = {}
    config_map.update({attr.upper(): getattr(config, attr) for attr in dir(config) if
                       not attr.startswith('__') and not callable(getattr(config, attr))})

    # then, update the configurations from the command line
    if parser:
        args = parser.parse_args()

        # update the config from the command line
        config_map.update({attr.upper(): getattr(args, attr) for attr in dir(args) if
                           not attr.startswith('__') and not callable(getattr(args, attr))})

    return config_map


def pop_up_image(batch_images):
    """
    Input shape: (batch_size, 3, height, width)
    """

    batch_images = batch_images.permute(0, 2, 3, 1)
    batch_images_npy = batch_images.detach().cpu().numpy()

    batch_images_npy = (batch_images_npy - batch_images_npy.min(axis=(1, 2, 3), keepdims=True)) / \
                       (batch_images_npy.max(axis=(1, 2, 3), keepdims=True) - batch_images_npy.min(axis=(1, 2, 3),
                                                                                                   keepdims=True))

    batch_images_npy = np.round(batch_images_npy * 255).astype(np.uint8)

    batch_size, height, width, _ = batch_images_npy.shape

    ncols = int(np.ceil(np.sqrt(batch_size)))
    nrows = int(np.ceil(batch_size / ncols))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2), squeeze=False)

    if nrows == 1 and ncols == 1:
        ax.imshow(batch_images_npy[0])
    else:
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx < batch_size:
                    ax[i, j].imshow(batch_images_npy[idx])
                    ax[i, j].axis('off')
                else:
                    ax[i, j].axis('off')

    plt.show()


mse_loss = torch.nn.MSELoss(reduce=True)
bce_loss = nn.BCEWithLogitsLoss(reduce=True)

import importlib

def load_class_by_name(class_name: str):
    """
    Load a class by its name
    """
    if class_name is None:
        return None
    module_name, class_name = class_name.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module {module_name} not found")
    return class_


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
