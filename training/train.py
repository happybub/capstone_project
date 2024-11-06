import os

import torch
from torch import nn

from training.utils import mse_loss, bce_loss
from training.utils import get_config, pop_up_image
from dataset.dataloader import get_dataloader

from training_utils import construct_discriminator_from_config, construct_model_from_config
from training_utils import load_state_from_checkpoint, save_state_to_checkpoint

from training.logger import Logger


def train_epoch(model, dataloader_map, config, epoch, epochs_logger, batches_logger:Logger, mode='train'):
    net, optim, discriminator, discriminator_optim = model
    # use_dis = (discriminator is not None and discriminator_optim is not None)
    use_dis = False
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
        # update_gen = i % 5 == 0
        update_gen = True
        update_dis = not update_gen
        # get the host images
        images = images.to(device=device)

        # generate the secrets message
        batch = images.size(0)
        secret = torch.randint(0, 2, (batch, num_bits)).float().to(device=device)

        with torch.set_grad_enabled(mode == 'train'):
            if mode == 'train':
                optim.zero_grad()
                if use_dis:
                    discriminator_optim.zero_grad()

            # forward pass
            container_image, secret_image, discarded_shape = net(secret, images)

            if i == 0 and mode == 'train':
                pop_up_image(torch.stack((images[0], container_image[0]), dim=0))

            # the model contains the generator and discriminator
            if use_dis:
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
                if mode == 'train' and update_dis:
                    discriminator_optim.step()

            # with torch.no_grad():
            attacked_image = net.attack_image(container_image)
            recovered_secret, recovered_secret_image = net.reverse(attacked_image, sampled_shape=discarded_shape)
            bit_acc = (recovered_secret.round() == secret).float().mean()

            # train the generator on the stego loss
            image_loss = mse_loss(container_image, images)
            secret_loss = mse_loss(recovered_secret, secret)
            stego_loss = lambda_image_loss * image_loss + lambda_secret_loss * secret_loss

            if mode == 'train' and not use_dis:
                # Since the backward of the fool_loss and stego_loss share some parts of the computation graph,
                # we deal with them separately.
                stego_loss.backward()

            if use_dis:
                # train the generator on the discriminator loss
                fake_output = discriminator(container_image)
                fool_loss = bce_loss(fake_output, torch.ones_like(fake_output))

                if mode == 'train':
                    (fool_loss + stego_loss).backward()
                    # fool_loss.backward()


            if mode == 'train' and update_gen:
                optim.step()

        batches_logger.log('Batch', i).log('Size', images.size(0)).log('Mode', mode).log('Update Gen', update_gen).log('Update Dis', update_dis) \
                    .log('Image', image_loss.item()).log('Secret', secret_loss.item()).log('Total', stego_loss.item()).log('Acc', bit_acc.item())
        if use_dis:
            batches_logger.log('Real', real_loss.item()).log('Fake', fake_loss.item()).log('Fool', fool_loss.item())
        batches_logger.save()
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
        epochs_logger.log("Real Acc", real_true_num / len(dataloader.dataset)).log("Fake Acc", fake_true_num / len(dataloader.dataset))

    epochs_logger.save()
    if mode == 'val':
        print('\r', epochs_logger.format_log(compare=True))



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

    logs_path = str(config['LOGS_PATH'])
    os.makedirs(os.path.join(logs_path, plan_name), exist_ok=True)
    train_epochs_logger = Logger(os.path.join(logs_path, plan_name, 'train_logs.log'))
    val_epochs_logger = Logger(os.path.join(logs_path, plan_name, 'val_logs.log'))

    # get the save frequency
    save_freq = config['SAVE_FREQ']

    # if the model is saved, load the model
    load_state_from_checkpoint(model, checkpoints_path, plan_name, start_epoch)
    for epoch in range(start_epoch, end_epoch + 1):

        # continue the training
        # print("Training epoch: ", epoch)
        train_batches_logger = Logger(os.path.join(logs_path, plan_name, f'train_epoch_{epoch}.log'))
        train_epoch(model, dataloader_map, config, epoch, train_epochs_logger, train_batches_logger, mode='train')
        train_batches_logger.save_to_file()

        # validate the model
        # print("Validating epoch: ", epoch)
        val_batches_logger = Logger(os.path.join(logs_path, plan_name, f'val_epoch_{epoch}.log'))
        train_epoch(model, dataloader_map, config, epoch, val_epochs_logger, val_batches_logger, mode='val')

        # save the logs in one epoch

        # save the state dict
        if save_freq != -1 and (epoch - start_epoch) % save_freq == 0 and epoch != start_epoch:
            save_state_to_checkpoint(model, checkpoints_path, plan_name, epoch)

    # save the logs in all epochs
    train_epochs_logger.save_to_file()
    val_epochs_logger.save_to_file()




def validation(plan_name, epoch, config):
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    batch = config['VAL_BATCH_SIZE']
    num_bits = config['NUM_BITS']
    device = config['DEVICE']

    net = construct_model_from_config(config)
    net.to(device=device)
    load_state_from_checkpoint((net, None, None, None), checkpoints_path, plan_name, epoch)

    net.eval()

    secret = torch.randint(0, 2, (batch, num_bits)).float().to(device=device)

    dataloader = get_dataloader(config)['val']

    # generate the host images
    for i, images in enumerate(dataloader):
        images = images.to(device=device)
        pop_up_image(images)

        container_image, secret_image, sampled_shape = net(secret, images)
        attacked_image = net.attack_image(container_image)
        recovered_secret, recovered_secret_image = net.reverse(attacked_image, sampled_shape)

        pop_up_image(container_image)
        # pop_up_image(secret_image)
        # pop_up_image(attacked_image)
        # pop_up_image(recovered_secret_image)

        bit_acc = (recovered_secret.round() == secret).float().mean()
        print(f'Batch: #{i}, Bit accuracy: {bit_acc}')
        break


if __name__ == '__main__':
    config_map = get_config()
    print(config_map)

    # get the time in format yyyymmdd:HHMMSS
    # time_str = time.strftime("%y%m%d_%H%M%S")
    # name = time_str

    # plan_name = '50_gen_20_dis'
    # start_epoch = 0
    # end_epoch = 10
    # plan_name = '50_only_gen'
    # start_epoch = 50
    # end_epoch = 60
    # plan_name = '30_only_gen'
    # start_epoch = 0
    # end_epoch = 30
    # train(plan_name, start_epoch, end_epoch, config_map)
    validation('preserved', 50, config_map)
    # validation('50_only_gen', 50, config_map)
