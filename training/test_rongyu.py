import os
import platform
import sys
import time

import numpy as np
from matplotlib import pyplot as plt

# Add the parent directory of 'training' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from training.utils import mse_loss, bce_loss
from training.utils import get_config, pop_up_image
from dataset.dataloader import get_dataloader

from training_utils import construct_discriminator_from_config, construct_model_from_config
from training_utils import load_state_from_checkpoint, save_state_to_checkpoint, early_stopping
from modules.text_embedding import LinearTextEmbedding1

from training.logger import Logger


#from training.train import


def test(plan_name, epoch, config, plot_fig=False, nbits=round(112 * 112 / 3), mode='val'):
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    batch = config['VAL_BATCH_SIZE']
    num_bits = config['NUM_BITS']
    device = config['DEVICE']

    text_emb = LinearTextEmbedding1(nbits, 1, 112, 112)
    secret_message = torch.randint(0, 2, (batch, nbits)).float().to(device=device)
    secret = text_emb.forward(secret_message)

    net = construct_model_from_config(config)
    net.to(device=device)
    load_state_from_checkpoint((net, None, None, None), checkpoints_path, plan_name, epoch)

    net.eval()

    net.image_embedding.pop_up_process = plot_fig

    if mode == 'val':
        dataloader = get_dataloader(config)['val']
    elif mode == 'test':
        dataloader = get_dataloader(config)['test']
    else:
        raise (ValueError('Invalid mode'))

    bit_acc_list = []
    image_loss_list = []
    secret_loss_list = []

    # generate the host images
    with torch.no_grad():
        for i, images in enumerate(dataloader):
            net.image_embedding.pop_up_process = plot_fig

            images = images.to(device=device)

            (freq_host_image, secret_image, freq_container, discarded), container_image = net(secret, images)

            sample = torch.rand_like(discarded).to(device=device)
            net.attack.x_offset = net.attack.y_offset = 56
            attacked_image = net.attack(container_image).to(device=device)
            (freq_attacked_container, sample, r_freq_container, r_secret_image), r_secret = net.reverse(attacked_image,
                                                                                                        sample)
            r_secret_message = text_emb.forward(r_secret, rev=True)
            r_secret_message = torch.where(r_secret_message > 0.5, 1, 0)
            print(container_image[0].mean(), attacked_image[0].mean(), (container_image[0] - attacked_image[0]).mean())
            bit_acc = (r_secret_message == secret_message).float().mean()
            print(f'Batch: #{i}, Bit accuracy: {bit_acc}')
            image_loss = mse_loss(freq_host_image, freq_container)
            secret = secret.view(secret.size(0), -1)
            secret_loss = mse_loss(r_secret, secret)

            bit_acc_list.append(bit_acc.detach().cpu().numpy())
            image_loss_list.append(image_loss.detach().cpu().numpy())
            secret_loss_list.append(secret_loss.detach().cpu().numpy())

            if plot_fig:
                break

            # Clear cache to free up memory
            torch.cuda.empty_cache()

    print('Average bit acc: ', np.mean(bit_acc_list))
    print('Average image loss: ', np.mean(image_loss_list))
    print('Average secret loss: ', np.mean(secret_loss_list))


if __name__ == '__main__':
    config_map = get_config()
    print(config_map)

    # get the time in format yyyymmdd:HHMMSS
    time_str = time.strftime("%y%m%d_%H%M%S")

    # name = '241110_210355' # embed 64

    # name = '241110_161636' # embed=128

    # name = '241110_234551'

    # name = 'vit'

    # name = '241111_060753'

    # name = '241111_074229'

    # name = '241111_080338'

    # name = '241111_082229'

    # name = '20241111_msehighfreq0.70.91.1.1.3'

    name = '12_rdbtwonoise'
    print(name)
    torch.manual_seed(42)
    test(name, 49, config_map, plot_fig=False, nbits=1024, mode='val')
