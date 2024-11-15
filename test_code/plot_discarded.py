# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 14:38
# @Author  : Gan Liyifan
# @File    : plot_discarded.py
import numpy as np
import torch
from matplotlib import pyplot as plt

from dataset.dataloader import get_dataloader
from training.training_utils import construct_model_from_config, load_state_from_checkpoint
from training.utils import get_config


def plot_discarded(plan_name, epoch, config, plot_fig=False, mode='val'):
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    batch = config['VAL_BATCH_SIZE']
    num_bits = config['NUM_BITS']
    device = config['DEVICE']

    mask_position = torch.zeros(112 * 112, dtype=torch.bool)

    # Randomly choose n_bits positions to set to True
    mask_position[torch.randperm(112 * 112)[:num_bits]] = True

    net = construct_model_from_config(config)
    net.to(device=device)
    load_state_from_checkpoint((net, None, None, None), checkpoints_path, plan_name, epoch)

    net.eval()

    net.image_embedding.pop_up_process = plot_fig

    secret = torch.randint(0, 2, (batch, 112 * 112)).float().to(device=device)

    if mode == 'val':
        dataloader = get_dataloader(config)['val']
    elif mode == 'test':
        dataloader = get_dataloader(config)['test']
    else:
        raise (ValueError('Invalid mode'))

    discarded_list = []


    with torch.no_grad():
        for i, images in enumerate(dataloader):
            print(f"Batch: {i}")
            net.image_embedding.pop_up_process = plot_fig

            images = images.to(device=device)

            (freq_host_image, secret_image, freq_container, discarded), container_image = net(secret, images)

            discarded_list.append(discarded.detach().cpu().numpy())

    # Flatten the discarded_list to a 1D array
    discarded_flat = np.concatenate(discarded_list).ravel()

    # Plot the empirical distribution
    plt.figure(figsize=(10, 6))
    plt.hist(discarded_flat, bins=100, density=True, alpha=0.6, color='g')

    # Add title and labels
    plt.title('Empirical Distribution of Discarded List')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Show the plot
    plt.grid(True)
    plt.savefig(name + '_discarded.png')


if __name__ == '__main__':
    config_map = get_config()
    print(config_map)

    name = '14_discardedl2norm'
    epoch = 49

    plot_discarded(name, epoch, config_map, plot_fig=False, mode='val')
