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


def pop_up_image(image):
    """
    Input shape: (batch_size, 3, height, width)
    """
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image_npy = image.detach().cpu().numpy()

    # normalize the image
    image_npy = (image_npy - image_npy.min()) / (image_npy.max() - image_npy.min())

    # round the image
    image_npy = np.round(image_npy * 255).astype(np.uint8)

    # plot the image
    plt.imshow(image_npy)
    plt.show()


def mse_loss(a, b):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(a, b)
    return loss


def load_class_by_name(class_name: str):
    """
    Load a class by its name
    """
    import importlib

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
