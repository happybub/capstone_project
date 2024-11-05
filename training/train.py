import os
import time

import torch
from torch import nn

from utils import mse_loss, bce_loss
from utils import get_config, pop_up_image
from dataset.dataloader import get_dataloader

from training_utils import construct_discriminator_from_config, construct_model_from_config
from training_utils import load_state_from_checkpoint, save_state_to_checkpoint


def train_epoch(model, dataloader_map, config, epoch, mode='train'):
    net, optim, discriminator, discriminator_optim = model
    GAN = (discriminator is not None and discriminator_optim is not None)
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

    for i, images in enumerate(dataloader):
        # get the host images
        images = images.to(device=device)

        # generate the secrets message
        batch = images.size(0)
        secret = torch.randint(0, 2, (batch, num_bits)).float().to(device=device)

        with torch.set_grad_enabled(mode == 'train'):
            # forward pass
            container_image, secret_image, discarded_shape = net(secret, images)
            attacked_image = net.attack_image(container_image)
            recovered_secret, recovered_secret_image = net.reverse(attacked_image, sampled_shape=discarded_shape)
            bit_acc = (recovered_secret.round() == secret).float().mean()

            if i == 0:
                pop_up_image(images)
                # pop_up_image(secret_image)
                pop_up_image(container_image)
                # pop_up_image(attacked_image)
                # pop_up_image(recovered_secret_image)

            if GAN:
                # training the discriminator
                if mode == 'train':
                    discriminator_optim.zero_grad()

                # train the discriminator on the real data
                original_image = images.detach()
                real_output = discriminator(original_image)
                real_loss = bce_loss(real_output, torch.ones_like(real_output))

                # the real_output is a logit, so we need to convert it to probability
                real_true_num += (nn.Sigmoid()(real_output) > 0.5).float().sum().item()
                if mode == 'train':
                    real_loss.backward()

                # train the discriminator on the fake data
                fake_image = container_image.detach()
                fake_output = discriminator(fake_image)
                fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))

                # the fake_output is a logit, so we need to convert it to probability
                fake_true_num += (nn.Sigmoid()(fake_output) < 0.5).float().sum().item()

                if mode == 'train':
                    fake_loss.backward()

                # update the discriminator
                if mode == 'train':
                    discriminator_optim.step()

            # training the generator
            if mode == 'train':
                optim.zero_grad()

            # train the generator on the stego loss
            image_loss = mse_loss(container_image, images)
            # image_loss = 0
            secret_loss = mse_loss(recovered_secret, secret)
            stego_loss = lambda_image_loss * image_loss + lambda_secret_loss * secret_loss

            if mode == 'train' and not GAN:
                # Since the backward of the fool_loss and stego_loss share some parts of the computation graph,
                # we deal with them separately.
                stego_loss.backward()

            if GAN:
                # train the generator on the discriminator loss
                fake_output = discriminator(container_image)
                fool_loss = bce_loss(fake_output, torch.ones_like(fake_output))
                if mode == 'train':
                    (fool_loss + stego_loss).backward()

            if mode == 'train':
                optim.step()
                # pass

        print(f'Batch: #{i}, Mode: {mode}, , '
              f'Secret: {secret_loss.item()}, Total: {stego_loss.item()}, Acc: {bit_acc} ' +
              f'Real: {real_loss.item()}, Fake: {fake_loss.item()}, Fool: {fool_loss}' if GAN else '')

    print(
        f'Epoch: {epoch}, Real Acc: {real_true_num / len(dataloader.dataset)}, Fake Acc: {fake_true_num / len(dataloader.dataset)}')


def train(plan_name, start_epoch, end_epoch, config):
    # get the model
    net = construct_model_from_config(config)
    optim = torch.optim.Adam(net.parameters(),
                             lr=float(config['LEARNING_RATE']),
                             weight_decay=config['WEIGHT_DECAY'])

    # get the discriminator
    discriminator = construct_discriminator_from_config(config)
    discriminator_optim = None
    if discriminator is not None:
        discriminator_optim = torch.optim.Adam(discriminator.parameters(),
                                               lr=float(config['DISCRIMINATOR_LEARNING_RATE']),
                                               weight_decay=config['WEIGHT_DECAY'])

    net.to(device=config['DEVICE'])
    discriminator.to(device=config['DEVICE'])

    model = (net, optim, discriminator, discriminator_optim)

    # get the dataloader
    dataloader_map = get_dataloader(config)

    # create the dictionary for the training
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    os.makedirs(os.path.join(checkpoints_path, plan_name), exist_ok=True)

    # get the save frequency
    save_freq = config['SAVE_FREQ']

    # if the model is saved, load the model
    load_state_from_checkpoint(model, checkpoints_path, plan_name, start_epoch)
    for epoch in range(start_epoch, end_epoch + 1):

        # continue the training
        print("Training epoch: ", epoch)
        train_epoch(model, dataloader_map, config, epoch, mode='train')

        # validate the model
        print("Validating epoch: ", epoch)
        train_epoch(model, dataloader_map, config, epoch, mode='val')

        # save the state dict
        if save_freq != -1 and (epoch - start_epoch) % save_freq == 0 and epoch != start_epoch:
            save_state_to_checkpoint(model, checkpoints_path, plan_name, epoch)


def validation(config):
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    name = '50_only_gen'
    batch = config['VAL_BATCH_SIZE']
    num_bits = config['NUM_BITS']
    device = config['DEVICE']

    net = construct_model_from_config(config)
    net.to(device=device)
    load_state_from_checkpoint(net, None, checkpoints_path, name, 50)

    net.eval()

    secret = torch.randint(0, 2, (batch, num_bits)).float().to(device=device)

    dataloader = get_dataloader(config)['val']

    # generate the host images
    for i, images in enumerate(dataloader):
        images = images.to(device=device)
        pop_up_image(images)

        container_image, secret_image, _ = net(secret, images)
        attacked_image = net.attack_image(container_image)
        recovered_secret, recovered_secret_image = net.reverse(attacked_image)

        pop_up_image(container_image)
        pop_up_image(secret_image)
        pop_up_image(attacked_image)
        pop_up_image(recovered_secret_image)

        bit_acc = (recovered_secret.round() == secret).float().mean()
        print(f'Batch: #{i}, Bit accuracy: {bit_acc}')
        break


if __name__ == '__main__':
    config_map = get_config()
    print(config_map)

    # get the time in format yyyymmdd:HHMMSS
    # time_str = time.strftime("%y%m%d_%H%M%S")
    # name = time_str
    plan_name = '50_gen_20_dis'
    # name = 'test_gan_1'
    start_epoch = 0
    end_epoch = 30

    train(plan_name, start_epoch, end_epoch, config_map)
    # validation(config_map)
