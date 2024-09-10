import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
