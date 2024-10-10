# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 11:25
# @Author  : Gan Liyifan
# @File    : simulate_distribution.py
import os
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from dataset.dataloader import get_dataloader
from modules.attack import AttackModule
from modules.dwt import DWTModule
from modules.image_embedding import ImageEmbeddingModule
from modules.text_embedding import TextEmbeddingModule
from utils import load_class_by_name, get_config


class distributionSimulation(nn.Module):
    def __init__(self, text_embedding: TextEmbeddingModule, dwt: DWTModule, image_embedding: ImageEmbeddingModule,
                 attack: AttackModule):
        super(distributionSimulation, self).__init__()
        self.text_embedding = text_embedding
        self.dwt = dwt
        self.image_embedding = image_embedding
        self.attack = attack
        self.activation = nn.Sigmoid()

    def forward(self, text_bits, host_image):
        device = text_bits.device
        freq_host_image = self.dwt(host_image)
        secret_image = self.text_embedding(text_bits)
        secret_image = secret_image.to(device)
        freq_secret_image = self.dwt(secret_image)
        freq_container, freq_noise = self.image_embedding(freq_host_image, freq_secret_image)
        # container_image = self.dwt(freq_container, rev=True)
        # return container_image
        return freq_noise


def simulation_epoch(net, dataloader_map, config, mode='train'):
    print_time = config['PRINT_TIME']
    net.print_time = print_time
    if mode == 'train':
        dataloader = dataloader_map['train']
        net.train()
    elif mode == 'val':
        dataloader = dataloader_map['val']
        net.eval()
    else:
        raise ValueError('mode should be either train or val')

    device = config['DEVICE']
    num_bits = int(config['NUM_BITS'])
    noise_list = []
    for i, images in enumerate(dataloader):
        # get the host images
        images = images.to(device=device)

        # generate the secrets message
        batch = images.size(0)
        secret = torch.randint(0, 2, (batch, num_bits)).float().to(device=device)

        with torch.set_grad_enabled(mode == 'train'):
            freq_noise = net.forward(secret, images)
            noise_list.append(freq_noise.detach().cpu().numpy())

    return noise_list


def simulation(name, start_epoch, end_epoch, config):
    # load the config
    channels, image_height, image_width = int(config['CHANNELS']), int(config['IMAGE_HEIGHT']), int(config['IMAGE_WIDTH'])
    num_bits = config['NUM_BITS']
    device = config['DEVICE']

    # construct the modules
    text_embedding_module = load_class_by_name(config_map['TEXT_EMBEDDING_MODULE'])(num_bits, channels=1, width=image_width,
                                                                                    height=image_height)
    dwt = load_class_by_name(config_map['DWT_MODULE'])()
    image_embedding_module = load_class_by_name(config_map['IMAGE_EMBEDDING_MODULE'])(channels, image_height,
                                                                                      image_width)
    attack_module = load_class_by_name(config_map['ATTACK_MODULE'])()

    # construct the model
    net = distributionSimulation(text_embedding_module, dwt, image_embedding_module, attack_module).to(device=device)

    # create the dictionary for the training
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    os.makedirs(os.path.join(checkpoints_path, name), exist_ok=True)

    # get the dataloader
    dataloader_map = get_dataloader(config)
    noise_list = []
    for epoch in range(start_epoch, end_epoch):
        print(f"Processing from epoch {epoch}")

        # continue the training
        noise_list_train = simulation_epoch(net, dataloader_map, config, mode='train')

        # validate the model
        noise_list_val = simulation_epoch(net, dataloader_map, config, mode='val')

        noise_list.extend(noise_list_train)
        noise_list.extend(noise_list_val)

    return noise_list

def present_distribution(freq_noise):
    freq_noise = freq_noise.detach().cpu().numpy()
    all_data = np.concatenate(freq_noise, axis=0)
    print(f"mean: {all_data.mean()}, std: {all_data.std()}")
    counts, bin_edges = np.histogram(all_data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], alpha=0.7, label='Empirical Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Empirical Distribution of Frequency Noise')
    plt.legend()
    plt.show()
    return all_data.mean(), all_data.std()


if __name__ == '__main__':
    config_map = get_config()
    print(config_map)

    time_str = time.strftime("%y%m%d_%H%M%S")
    name = time_str
    start_epoch = 1
    end_epoch = 5

    noise_list = simulation(time_str, start_epoch, end_epoch, config_map)
    print(len(noise_list))
    print(noise_list[0].shape)

    all_data = np.concatenate([np.array(arr).flatten() for arr in noise_list])
    plt.figure(figsize=(10, 6))
    plt.hist(all_data, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Elements in noise_list')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    mean_value = np.mean(all_data)
    std_value = np.std(all_data)
    min_value = np.min(all_data)
    max_value = np.max(all_data)

    print(f'Mean: {mean_value}')
    print(f'Standard Deviation: {std_value}')
    print(f'Min: {min_value}')
    print(f'Max: {max_value}')

    # simulate the data using poisson distribution
    data = np.array(all_data)

    transformed_data = (data + 1) * 2

    lambda_est = np.mean(transformed_data)

    poisson_dist = stats.poisson(mu=lambda_est)
    bins = np.arange(transformed_data.min(), transformed_data.max() + 1)
    counts, _ = np.histogram(transformed_data, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.bar(bin_centers / 2 - 1, counts, width=0.5, alpha=0.5, label='Empirical Data')

    x = np.arange(transformed_data.min(), transformed_data.max() + 1)
    plt.plot(x / 2 - 1, poisson_dist.pmf(x), 'r-', label=f'Poisson Fit (λ={lambda_est:.2f})')

    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
