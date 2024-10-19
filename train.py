import json
import os
import time

import torch

from utils import mse_loss
from utils import load_class_by_name, get_config, pop_up_image
from dataset.dataloader import get_dataloader
from modules.model import OurModel


def train_epoch(net, optim, dataloader_map, config, epoch, mode='train', noise_logs=[]):
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

    losses = {
        'image_losses': [],
        'secret_losses': [],
        'total_losses': [],
        'bit_acc': []
    }

    for i, images in enumerate(dataloader):
        # get the host images
        images = images.to(device=device)

        # generate the secrets message
        batch = images.size(0)
        secret = torch.randint(0, 2, (batch, num_bits)).float().to(device=device)

        with torch.set_grad_enabled(mode == 'train'):

            # forward pass
            container_image, extracted_secret = net(secret, images, return_extracted_secret=True)

            # attack the images
            attacked_image = net.attack_image(container_image)

            # recover the secret message
            recovered_secret, sampled_secret = net.reverse(attacked_image, extracted_secret=extracted_secret)
            if i == 0:
                noise_logs.append((extracted_secret[0], sampled_secret[0]))

            # to test, we directly use the oringal text as the recovered text
            # recovered_secret = secret

            # calculate the loss
            image_loss = mse_loss(container_image, images)
            secret_loss = mse_loss(recovered_secret, secret)

            total_loss = lambda_image_loss * image_loss + lambda_secret_loss * secret_loss
            bit_acc = (recovered_secret.round() == secret).float().mean()

            # backward pass
            if mode == 'train':
                optim.zero_grad()
                total_loss.backward()
                optim.step()

        losses['image_losses'].append(image_loss.item())
        losses['secret_losses'].append(secret_loss.item())
        losses['total_losses'].append(total_loss.item())
        losses['bit_acc'].append(bit_acc.item())
        print(f'Batch: #{i}, Mode: {mode}, Image Loss: {image_loss.item()}, '
              f'Secret Loss: {secret_loss.item()}, Total Loss: {total_loss}, Bit accuracy: {bit_acc}')

    # log_dir = config['LOG_DIR']
    # # create the log file
    # os.makedirs(os.path.join(log_dir, f'mode: {mode} epoch: {epoch}'), exist_ok=True)
    # with open(os.path.join(log_dir, f'mode: {mode} epoch: {epoch}', 'image_loss_log.txt'), 'w') as f:
    #     f.write('\n'.join([str(item) for item in image_losses]))
    # with open(os.path.join(log_dir, f'mode: {mode} epoch: {epoch}', 'secret_loss_log.txt'), 'w') as f:
    #     f.write('\n'.join([str(item) for item in secret_losses]))

    return noise_logs, losses


def train(name, start_epoch, end_epoch, config):
    net = construct_model_from_config(config)
    optim = torch.optim.Adam(net.parameters(), lr=float(config['LEARNING_RATE']), weight_decay=config['WEIGHT_DECAY'])

    # get the dataloader
    dataloader_map = get_dataloader(config)

    # create the dictionary for the training
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    os.makedirs(os.path.join(checkpoints_path, name), exist_ok=True)

    # get the save frequency
    save_freq = config['SAVE_FREQ']

    # for debugging
    noise_logs = []

    # for logs
    log_dir = config['LOG_DIR']
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(start_epoch, end_epoch + 1):
        # if the model is saved, load the model
        load_state_from_checkpoint(net, optim, checkpoints_path, name, epoch, config['DEVICE'])

        # continue the training
        print("Training epoch: ", epoch)
        noise_logs, losses = train_epoch(net, optim, dataloader_map, config, epoch, mode='train', noise_logs=noise_logs)

        # validate the model
        print("Validating epoch: ", epoch)
        _, losses_valid = train_epoch(net, optim, dataloader_map, config, epoch, mode='val')

        # save the logs
        log_file_path = os.path.join(log_dir, 'training_logs.json')
        log_data = {
            'epoch': epoch,
            'train_losses': losses,
            'val_losses': losses_valid
        }
        with open(log_file_path, 'a') as log_file:
            log_file.write(json.dumps(log_data) + '\n')

        # save the state dict
        if save_freq != -1 and epoch % save_freq == 0:
            save_state_to_checkpoint(net, optim, checkpoints_path, name, epoch)

    pop_up_image(torch.stack([p[0] for p in noise_logs], dim=0))
    pop_up_image(torch.stack([p[1] for p in noise_logs], dim=0))


def construct_model_from_config(config):
    channels, image_height, image_width = int(config['CHANNELS']), int(config['IMAGE_HEIGHT']), int(
        config['IMAGE_WIDTH'])
    num_bits = config['NUM_BITS']
    device = config['DEVICE']

    # construct the modules
    text_embedding_module = load_class_by_name(config_map['TEXT_EMBEDDING_MODULE'])(num_bits, channels=1,
                                                                                    width=image_width,
                                                                                    height=image_height)
    dwt = load_class_by_name(config_map['DWT_MODULE'])()
    image_embedding_module = load_class_by_name(config_map['IMAGE_EMBEDDING_MODULE'])(channels, image_height,
                                                                                      image_width)
    attack_module = load_class_by_name(config_map['ATTACK_MODULE'])()

    # construct the model
    net = OurModel(text_embedding_module, dwt, image_embedding_module, attack_module).to(device=device)
    return net


def load_state_from_checkpoint(net, optim, checkpoints_path, name: str, epoch, device):
    expected_model_path = os.path.join(checkpoints_path, name, f'{epoch}.pth')
    if epoch > 0 and os.path.exists(expected_model_path):
        # net.load_state_dict(torch.load(expected_model_path))
        net.load_state_dict(torch.load(expected_model_path, map_location=torch.device(device)))
        print(f'Load the model from {epoch}.pth')
    else:
        print(f'No model found in {epoch}.pth')

    if optim is not None:
        expected_optim_path = os.path.join(checkpoints_path, name, f'{epoch}_optim.pth')
        if epoch > 0 and os.path.exists(expected_optim_path):
            # optim.load_state_dict(torch.load(expected_optim_path))
            optim.load_state_dict(torch.load(expected_optim_path, map_location=torch.device(device)))
            print(f'Load the optimizer from {epoch}_optim.pth')
        else:
            print(f'No optimizer found in {epoch}_optim.pth')


def save_state_to_checkpoint(net, optim, checkpoints_path, name: str, epoch):
    torch.save(net.state_dict(), os.path.join(checkpoints_path, name, f'{epoch}.pth'))
    torch.save(optim.state_dict(), os.path.join(checkpoints_path, name, f'{epoch}_optim.pth'))


def validation(config):
    net = construct_model_from_config(config)

    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    name = '241018_191855'
    batch = config['VAL_BATCH_SIZE']
    num_bits = config['NUM_BITS']
    device = config['DEVICE']
    load_state_from_checkpoint(net, None, checkpoints_path, name, 60, device)

    net.eval()

    secret = torch.randint(0, 2, (batch, num_bits)).float().to(device=device)

    dataloader = get_dataloader(config)['val']

    # generate the host images
    for i, images in enumerate(dataloader):
        images = images.to(device=device)
        pop_up_image(images)

        container_image, extracted_secret = net(secret, images, return_extracted_secret=True)
        pop_up_image(container_image)
        pop_up_image(extracted_secret)

        attacked_image = net.attack_image(container_image)

        recovered_secret, sampled_secret = net.reverse(attacked_image, extracted_secret=extracted_secret)
        pop_up_image(sampled_secret)

        bit_acc = (recovered_secret.round() == secret).float().mean()
        print(f'Batch: #{i}, Bit accuracy: {bit_acc}')
        break









if __name__ == '__main__':
    config_map = get_config()
    print(config_map)

    # get the time in format yyyymmdd:HHMMSS
    time_str = time.strftime("%y%m%d_%H%M%S")
    name = time_str
    start_epoch = 1
    end_epoch = 70

    train(name, start_epoch, end_epoch, config_map)
    validation(config_map)
