import os

import torch

from modules.model import OurModel
from training.utils import load_class_by_name


def construct_model_from_config(config):
    channels, image_height, image_width = int(config['CHANNELS']), int(config['IMAGE_HEIGHT']), int(
        config['IMAGE_WIDTH'])
    num_bits = config['NUM_BITS']

    # construct the modules
    text_embedding_module = load_class_by_name(config['TEXT_EMBEDDING_MODULE'])(num_bits, channels=1,
                                                                                    width=image_width,
                                                                                    height=image_height)
    dwt = load_class_by_name(config['DWT_MODULE'])()
    image_embedding_module = load_class_by_name(config['IMAGE_EMBEDDING_MODULE'])(channels, image_height,
                                                                                      image_width)
    attack_module = load_class_by_name(config['ATTACK_MODULE'])()

    # construct the model
    net = OurModel(text_embedding_module, dwt, image_embedding_module, attack_module)

    return net


def construct_discriminator_from_config(config):
    discriminator = load_class_by_name(config['DISCRIMINATOR_MODULE'])(input_channels=int(config['DISCRIMINATOR_INPUT_CHANNELS']))
    return discriminator


def load_state_from_checkpoint(model, checkpoints_path, plan_name: str, epoch):
    net, optim, discriminator, discriminator_optim = model
    models_info = [
        ('model', f'{epoch}.pth', net),
        ('optim', f'{epoch}_optim.pth', optim),
        ('discriminator', f'{epoch}_discriminator.pth', discriminator),
        ('discriminator_optim', f'{epoch}_discriminator_optim.pth', discriminator_optim)
    ]
    os.makedirs(os.path.join(checkpoints_path, plan_name), exist_ok=True)
    for name, file_name, module in models_info:
        if module is None:
            print(f'{name} is None')
            continue
        path = os.path.join(checkpoints_path, plan_name, file_name)
        if os.path.exists(path):
            module.load_state_dict(torch.load(path))
            print(f'Load the {name} from {epoch}.pth')
        else:
            print(f'No {name} found in {epoch}.pth')


def save_state_to_checkpoint(model, checkpoints_path, plan_name: str, epoch):
    net, optim, discriminator, discriminator_optim = model
    models_info = [
        ('model', f'{epoch}.pth', net),
        ('optim', f'{epoch}_optim.pth', optim),
        ('discriminator', f'{epoch}_discriminator.pth', discriminator),
        ('discriminator_optim', f'{epoch}_discriminator_optim.pth', discriminator_optim)
    ]
    os.makedirs(os.path.join(checkpoints_path, plan_name), exist_ok=True)
    for name, file_name, module in models_info:
        if module is None:
            print(f'{name} is None')
            continue
        path = os.path.join(checkpoints_path, plan_name, file_name)
        torch.save(module.state_dict(), path)
        print(f'Save the {name} to {epoch}.pth')


def early_stopping(val_loss, patience=3):
    if len(val_loss) < patience or len(val_loss) < 10:
        return False
    for i in range(1, patience):
        if val_loss[-i] < val_loss[-i - 1]:
            return False
    return True