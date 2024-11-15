import os
import platform
import sys
import time

# Add the parent directory of 'training' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import StegoDataset


import torch

from training.utils import mse_loss, bce_loss
from training.utils import get_config, pop_up_image
from dataset.dataloader import get_dataloader

from training_utils import construct_discriminator_from_config, construct_model_from_config
from training_utils import load_state_from_checkpoint, save_state_to_checkpoint, early_stopping

from training.logger import Logger


def train_epoch(model, dataloader_map, config, epoch, epochs_logger, batches_logger: Logger, mode='train'):
    net, optim, discriminator, discriminator_optim = model
    # discriminator = None
    use_dis = (discriminator is not None and discriminator_optim is not None)
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
    lambda_image_loss = float(config['LAMBDA_IMAGE_LOSS'])
    lambda_secret_loss = float(config['LAMBDA_SECRET_LOSS'])

    real_true_num = 0
    fake_true_num = 0
    update_gen = True
    update_dis = True

    mask_position = torch.zeros(112 * 112, dtype=torch.bool)

    # Randomly choose n_bits positions to set to True
    mask_position[torch.randperm(112 * 112)[:num_bits]] = True


    for i, images in enumerate(dataloader):
        # get the host images
        images = images.to(device=device)

        # generate the secrets message
        batch = images.size(0)
        secret = torch.randint(0, 2, (batch, 112 * 112)).float().to(device=device)

        with torch.set_grad_enabled(mode == 'train'):
            if mode == 'train':
                optim.zero_grad()
                if use_dis:
                    discriminator_optim.zero_grad()

            # forward pass
            (freq_host_image, secret_image, freq_container, discarded), container_image = net(secret, images)
            sample = torch.rand_like(discarded).to(device=device)

            # all zero
            # sample = torch.zeros(discarded.shape).to(device=device)

            # universe dist.
            # sample = torch.rand(discarded.shape).to(device=device)

            # the model contains the generator and discriminator
            if use_dis:
                # train the discriminator on the fake data
                fake_image = discarded.detach()
                fake_output = discriminator(fake_image)
                fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))

                # the fake_output is a logit, so we need to convert it to probability
                fake_true_num += (fake_output < 0.5).float().sum().item()

                if mode == 'train':
                    fake_loss.backward()

                # train the discriminator on the real data
                original_image = sample
                real_output = discriminator(original_image)
                real_loss = bce_loss(real_output, torch.ones_like(real_output))

                real_true_num += (real_output > 0.5).float().sum().item()
                if mode == 'train':
                    real_loss.backward()

                # update the discriminator
                if mode == 'train' and update_dis:
                    discriminator_optim.step()

            attacked_image = net.attack_image(container_image).to(device=device)

            (freq_attacked_container, sample, r_freq_container, r_secret_image), r_secret = net.reverse(attacked_image, sample)
            bit_acc = (r_secret[:, mask_position].round() == secret[:, mask_position]).float().mean()

            # train the generator on the stego loss

            # freq_host_img_1 = freq_host_image[:, 0:3]
            # freq_host_img_2 = freq_host_image[:, 3:6]
            # freq_host_img_3 = freq_host_image[:, 6:9]
            # freq_host_img_4 = freq_host_image[:, 9:12]
            #
            # freq_container_1 = freq_container[:, 0:3]
            # freq_container_2 = freq_container[:, 3:6]
            # freq_container_3 = freq_container[:, 6:9]
            # freq_container_4 = freq_container[:, 9:12]
            #
            # image_loss_1 = mse_loss(freq_host_img_1, freq_container_1)
            # image_loss_2 = mse_loss(freq_host_img_2, freq_container_2)
            # image_loss_3 = mse_loss(freq_host_img_3, freq_container_3)
            # image_loss_4 = mse_loss(freq_host_img_4, freq_container_4)
            #
            # freq_loss = image_loss_2
            image_loss = mse_loss(freq_host_image, freq_container)
            secret_loss = mse_loss(r_secret[:, mask_position], secret[:, mask_position])

            discarded_l2_norm = torch.norm(discarded, p=2)
            stego_loss = lambda_image_loss * image_loss + lambda_secret_loss * secret_loss + discarded_l2_norm
            # stego_loss = lambda_image_loss * freq_loss + lambda_secret_loss * secret_loss

            if mode == 'train' and not use_dis:
                # Since the backward of the fool_loss and stego_loss share some parts of the computation graph,
                # we deal with them separately.
                stego_loss.backward()

            if use_dis:
                # train the generator on the discriminator loss
                fake_image = discarded
                fake_output = discriminator(fake_image)
                fool_loss = bce_loss(fake_output, torch.ones_like(fake_output))

                if mode == 'train':
                    (fool_loss + stego_loss).backward()
                    # fool_loss.backward()

            if mode == 'train' and update_gen:
                optim.step()

        (batches_logger.log('Batch', i).log('Size', images.size(0)).log('Mode', mode).log('Update Gen', update_gen).log(
            'Update Dis', update_dis) \
            .log('Image', image_loss.item()).log('Secret', secret_loss.item()) \
            .log('Total', stego_loss.item()).log('Acc', bit_acc.item()))
        if use_dis:
            batches_logger.log('Real', real_loss.item()).log('Fake', fake_loss.item()).log('Fool', fool_loss.item())
        batches_logger.save()
        if platform.system() == 'Linux':
            print(batches_logger.format_log(compare=True), end='\r')
        else:
            print('\r', batches_logger.format_log(compare=True), end='')

    epochs_logger.log('Epoch', epoch).log('Mode', mode)
    epochs_logger.log("Image", batches_logger.get_values_mean('Image')) \
        .log("Secret", batches_logger.get_values_mean('Secret')) \
        .log("Total", batches_logger.get_values_mean('Total')) \
        .log("Acc", batches_logger.get_values_mean('Acc'))
    if use_dis:
        epochs_logger.log("Real", batches_logger.get_values_mean('Real')) \
            .log("Fake", batches_logger.get_values_mean('Fake')) \
            .log("Fool", batches_logger.get_values_mean('Fool'))
        epochs_logger.log("Real Acc", real_true_num / len(dataloader.dataset)).log("Fake Acc", fake_true_num / len(
            dataloader.dataset))

    epochs_logger.save()
    if mode == 'val':
        if platform.system() == 'Linux':
            print(epochs_logger.format_log(compare=True))
        else:
            print('\r', epochs_logger.format_log(compare=True))


def train(based_name, start_epoch, plan_name, end_epoch, config):
    # get the model
    net = construct_model_from_config(config)
    # optim = torch.optim.Adam([{'params': net.text_embedding.parameters(), 'lr': 1e-3},
    #                           {'params': net.image_embedding.parameters(), 'lr': config['LEARNING_RATE']}],
    #                          weight_decay=config['WEIGHT_DECAY'])
    optim = torch.optim.Adam(net.parameters(), lr=float(config['LEARNING_RATE']), weight_decay=config['WEIGHT_DECAY'])
    # get the discriminator
    discriminator = construct_discriminator_from_config(config)
    discriminator = None
    discriminator_optim = None
    if discriminator is not None:
        discriminator_optim = torch.optim.Adam(discriminator.parameters(),
                                               lr=float(config['DISCRIMINATOR_LEARNING_RATE']),
                                               weight_decay=config['WEIGHT_DECAY'])

    net.to(device=config['DEVICE'])
    if discriminator is not None:
        discriminator.to(device=config['DEVICE'])

    model = (net, optim, discriminator, discriminator_optim)

    # get the dataloader
    dataloader_map = get_dataloader(config)

    # check for existing dictionary
    duplicated_count = 0
    if os.path.exists(os.path.join(str(config['CHECKPOINTS_PATH']), plan_name)) and based_name != plan_name:
        duplicated_count += 1
    while os.path.exists(os.path.join(str(config['CHECKPOINTS_PATH']),
                                      plan_name + f'_{duplicated_count}')) and based_name != plan_name:
        duplicated_count += 1
    if duplicated_count > 0:
        plan_name = plan_name + f'_{duplicated_count}'

    # create the dictionary for the training
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    os.makedirs(os.path.join(checkpoints_path, plan_name), exist_ok=True)

    logs_path = str(config['LOGS_PATH'])
    os.makedirs(os.path.join(logs_path, plan_name), exist_ok=True)
    train_epochs_logger = Logger(os.path.join(logs_path, plan_name, 'train_logs.log'))
    val_epochs_logger = Logger(os.path.join(logs_path, plan_name, 'val_logs.log'))

    # get the save frequency
    save_freq = config['SAVE_FREQ']

    # if the model is saved, load the model
    load_state_from_checkpoint(model, checkpoints_path, based_name, start_epoch - 1)
    for epoch in range(start_epoch, end_epoch + 1):
        # continue the training
        # print("Training epoch: ", epoch)
        train_batches_logger = Logger(os.path.join(logs_path, plan_name, f'train_epoch_{epoch}.log'))
        train_epoch(model, dataloader_map, config, epoch, train_epochs_logger, train_batches_logger, mode='train')
        train_batches_logger.save_to_file()

        # validate the model
        # print("Validating epoch: ", epoch)
        # net.image_embedding.pop_up_process = True
        val_batches_logger = Logger(os.path.join(logs_path, plan_name, f'val_epoch_{epoch}.log'))
        train_epoch(model, dataloader_map, config, epoch, val_epochs_logger, val_batches_logger, mode='val')

        # save the logs in one epoch

        # save the state dict
        if save_freq != -1 and (epoch - start_epoch + 1) % save_freq == 0 and epoch != start_epoch:
            save_state_to_checkpoint(model, checkpoints_path, plan_name, epoch)

            train_epochs_logger.save_to_file()
            val_epochs_logger.save_to_file()

        val_losses = val_epochs_logger.get_values('Total')
        # if early_stopping(val_losses):
        #     print('Early stopping')
        #     break

    # save the logs in all epochs
    train_epochs_logger.save_to_file()
    val_epochs_logger.save_to_file()


# def validation(plan_name, epoch, config, plot_fig=False):
#     checkpoints_path = str(config['CHECKPOINTS_PATH'])
#     batch = config['VAL_BATCH_SIZE']
#     num_bits = config['NUM_BITS']
#     device = config['DEVICE']
#
#     mask_position = torch.zeros(112 * 112, dtype=torch.bool)
#
#     # Randomly choose n_bits positions to set to True
#     mask_position[torch.randperm(112 * 112)[:num_bits]] = True
#
#     net = construct_model_from_config(config)
#     net.to(device=device)
#     load_state_from_checkpoint((net, None, None, None), checkpoints_path, plan_name, epoch)
#
#     net.eval()
#
#     net.image_embedding.pop_up_process = plot_fig
#
#     secret = torch.randint(0, 2, (batch, 112 * 112)).float().to(device=device)
#
#     dataloader = get_dataloader(config)['val']
#
#     bit_acc_list = []
#     image_loss_list = []
#     secret_loss_list = []
#
#     # generate the host images
#     for i, images in enumerate(dataloader):
#         net.image_embedding.pop_up_process = plot_fig
#
#         images = images.to(device=device)
#
#         (freq_host_image, secret_image, freq_container, discarded), container_image = net(secret, images)
#
#         sample = torch.rand_like(discarded).to(device=device)
#         net.attack.x_offset = net.attack.y_offset = 56
#         attacked_image = net.attack(container_image).to(device=device)
#         (freq_attacked_container, sample, r_freq_container, r_secret_image), r_secret = net.reverse(attacked_image, sample)
#
#         if plot_fig:
#             log_plt_path = "../log_plots/"
#             model_name = name + '/'
#             os.makedirs("log_plt_path", exist_ok=True)
#             os.makedirs(log_plt_path + model_name, exist_ok=True)
#             for jj in range(0, 4):
#                 fig = freq_container[0][jj * 3: jj * 3 + 3] - freq_host_image[0][jj * 3: jj * 3 + 3]
#             fig = 1 - fig
#             fig = fig.permute(1, 2, 0).detach().numpy()
#             plt.imshow(fig)
#             plt.savefig(log_plt_path + model_name + '__' + str(jj) + '.png', format='png', dpi=300)
#             # plt.show()
#
#         if plot_fig:
#             pop_up_image([images[0], container_image[0], images[0] - container_image[0]])
#
#         print(container_image[0].mean(), attacked_image[0].mean(), (container_image[0] - attacked_image[0]).mean())
#         bit_acc = (r_secret[:, mask_position].round() == secret[:, mask_position]).float().mean()
#         print(f'Batch: #{i}, Bit accuracy: {bit_acc}')
#         image_loss = mse_loss(freq_host_image, freq_container)
#         secret_loss = mse_loss(r_secret[:, mask_position], secret[:, mask_position])
#
#         bit_acc_list.append(bit_acc)
#         image_loss_list.append(image_loss)
#         secret_loss_list.append(secret_loss)
#
#         if plot_fig:
#             break
#
#     print('Average bit acc: ', np.mean(bit_acc_list))
#     print('Average image loss: ', np.mean(image_loss_list))
#     print('Average secret loss: ', np.mean(secret_loss_list))

def validation(plan_name, epoch, config, plot_fig=False, mode = 'val'):
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

    if plot_fig:
        image_path = '0801.png'
        imagedata = StegoDataset('../data/test/DIV2K_valid_HR')
        imagedata.images_path = [image_path]
        dataloader = DataLoader(imagedata, batch_size=1, shuffle=False, num_workers=0)
    else:
        if mode == 'val':
            dataloader = get_dataloader(config)['val']
        elif mode == 'test':
            dataloader = get_dataloader(config)['test']
        else:
            raise (ValueError('Invalid mode'))

    bit_acc_list = []
    image_loss_list = []
    secret_loss_list = []
    image_psnr_list = []

    with torch.no_grad():
        for i, images in enumerate(dataloader):
            net.image_embedding.pop_up_process = plot_fig

            images = images.to(device=device)

            (freq_host_image, secret_image, freq_container, discarded), container_image = net(secret, images)

            sample = torch.rand_like(discarded).to(device=device)

            image_psnr = calculate_psnr(images, container_image)
            image_psnr_list.append(image_psnr)

            # sample = torch.rand(discarded.shape).to(device=device)

            net.attack.x_offset = net.attack.y_offset = 56
            attacked_image = net.attack(container_image).to(device=device)
            (freq_attacked_container, sample, r_freq_container, r_secret_image), r_secret = net.reverse(attacked_image, sample)

            # if plot_fig:
            #     log_plt_path = "../log_plots/"
            #     model_name = name + '/'
            #     os.makedirs("log_plt_path", exist_ok=True)
            #     os.makedirs(log_plt_path + model_name, exist_ok=True)
            #     for jj in range(0, 4):
            #         [...]
            #     fig = 1 - fig
            #     fig = fig.permute(1, 2, 0).detach().numpy()
            #     plt.imshow(fig)
            #     plt.savefig(log_plt_path + model_name + '__' + str(jj) + '.png', format='png', dpi=300)
            #     # plt.show()

            if plot_fig:
                pop_up_image([images[0], container_image[0], images[0] - container_image[0]])

            print(container_image[0].mean(), attacked_image[0].mean(), (container_image[0] - attacked_image[0]).mean())
            bit_acc = (r_secret[:, mask_position].round() == secret[:, mask_position]).float().mean()
            print(f'Batch: #{i}, Bit accuracy: {bit_acc}')
            image_loss = mse_loss(freq_host_image, freq_container)
            secret_loss = mse_loss(r_secret[:, mask_position], secret[:, mask_position])

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
    print('Average PSNR: ', np.mean(image_psnr_list))

def calculate_psnr(img1, img2):
    import math
    # img1 and img2 have range [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

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

    name = '15_l2normre'
    print(name)
    torch.manual_seed(42)
    train(name, 0, name, 50, config_map)

    validation(name, 49, config_map, plot_fig=True, mode='val')
    # validation(name, 30, config_map)
    # validation('50_only_gen', 50, config_map)
