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


def pop_up_image(images, display_size=(112, 112)):
    """
    Display images where inputs can be:
    - A single tensor
    - A list of tensors
    - A list of lists, each containing tensors

    Args:
        images: single tensor, list of tensors, or list of lists of tensors
        display_size (tuple): the target display size (height, width) for each image.
    """

    def process_image(image):
        image = image.detach().cpu()
        """Process and normalize a single image tensor."""
        if image.ndim == 2:  # H, W -> add channel dimension
            image = image.unsqueeze(0)  # C=1
            ori_size = image.shape
            image = torch.nn.functional.interpolate(image, size=display_size, mode='bilinear', align_corners=False)
            images = [(image, ori_size)]
        elif image.ndim == 3:  # C, H, W
            ori_size = image.shape
            image = torch.nn.functional.interpolate(image.unsqueeze(0), size=display_size, mode='bilinear',
                                                    align_corners=False).squeeze(0)
            images = [(image, ori_size)]
        elif image.ndim == 4:  # B, C, H, W
            ori_size = image.shape[1:]
            image = torch.nn.functional.interpolate(image, size=display_size, mode='bilinear', align_corners=False)
            images = [(image[i], ori_size) for i in range(image.shape[0])]
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

        # image = (image * 255).byte()
        return [((image - image.min()) / (image.max() - image.min()), ori_size) for image, ori_size in images]

    def flatten_and_process(images):
        """Flatten and process images to ensure they are a list of lists of tensors."""
        if isinstance(images, torch.Tensor):
            images = [images]

        ret = []
        if isinstance(images, list) and all(isinstance(item, torch.Tensor) for item in images):
            processed = []
            for item in images:
                processed.extend(process_image(item))
            ret.append(processed)
            return ret

        if isinstance(images, list) and all(isinstance(item, list) for item in images):
            for row in images:
                processed = []
                for item in row:
                    processed.extend(process_image(item))
                ret.append(processed)
            return ret

        raise ValueError(f"Invalid input type: {type(images)}")

    processed_images = flatten_and_process(images)

    # Determine number of rows and columns
    nrows = len(processed_images)
    ncols = max(len(item) for item in processed_images)

    # Create subplots and display images
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3), squeeze=False)

    for i, row in enumerate(processed_images):
        for j, (image, ori_size) in enumerate(row):
            # Assuming image is a tensor of shape (C, H, W)
            if image.shape[0] == 1:
                # axes[i, j].imshow(image[0].numpy(), cmap='gray')
                axes[i, j].imshow(image[0].numpy())
            else:  # RGB image
                axes[i, j].imshow(image.permute(1, 2, 0).numpy())

            axes[i, j].axis('off')

            # Display the shape of the tensor above the image
            label = f'Size: {ori_size[0]}x{ori_size[1]}x{ori_size[2]}' if len(ori_size) == 3 else f'Size: {ori_size[0]}x{ori_size[1]}'

            # Place the text above the image, centered horizontally
            axes[i, j].text(0.5, 1.05, label, color='black', fontsize=12, ha='center',
                            va='bottom', transform=axes[i, j].transAxes)

    plt.show()

def pop_up_attention_map(attention_weights, query_id=None, patch_size=7, title=''):
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    if query_id is not None:
        # Extract the attention weights for the specific query
        attention_weights = attention_weights[query_id]

        # Create an image where each patch is colored based on the attention weight
        patch_grid_side = int(np.sqrt(len(attention_weights)))  # Assume square grid
        img_size = patch_size * patch_grid_side

        attention_image = np.zeros((img_size, img_size))

        for i in range(patch_grid_side):
            for j in range(patch_grid_side):
                # Calculate the intensity of the color based on the attention weight
                intensity = attention_weights[i * patch_grid_side + j]
                attention_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = intensity

        # Plot the attention image
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_image, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'{title} Attention Map for Query ID {query_id}')
        plt.axis('off')  # Hide the axes
        plt.show()

    else:
        # Use matplotlib to create a heatmap of the attention weights
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        plt.title(f'{title} Attention Map')
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
