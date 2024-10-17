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

    # 将批次中的每张图像从 (3, height, width) 转置为 (height, width, 3)
    batch_images = batch_images.permute(0, 2, 3, 1)

    # 将图像从张量转换为numpy数组
    batch_images_npy = batch_images.detach().cpu().numpy()

    # 归一化每个图像
    batch_images_npy = (batch_images_npy - batch_images_npy.min(axis=(1, 2, 3), keepdims=True)) / \
                       (batch_images_npy.max(axis=(1, 2, 3), keepdims=True) - batch_images_npy.min(axis=(1, 2, 3),
                                                                                                   keepdims=True))

    # 将归一化后的值转换为0-255的整数用于显示
    batch_images_npy = np.round(batch_images_npy * 255).astype(np.uint8)

    # 获取批次大小和图像尺寸
    batch_size, height, width, _ = batch_images_npy.shape

    # 创建一个足够大的画布显示所有图像
    ncols = int(np.ceil(np.sqrt(batch_size)))  # 确定列数
    nrows = int(np.ceil(batch_size / ncols))  # 确定行数

    # 设置画布大小
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2))

    # 如果只有一个图像，ax不会是数组
    if nrows == 1 and ncols == 1:
        ax.imshow(batch_images_npy[0])
    else:
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx < batch_size:
                    ax[i, j].imshow(batch_images_npy[idx])
                    ax[i, j].axis('off')  # 不显示坐标轴
                else:
                    ax[i, j].axis('off')  # 对于没有图像的部分也不显示坐标轴

    plt.show()


def mse_loss(a, b):
    loss_fn = torch.nn.MSELoss(reduce=True)
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
