import os

import torch

from utils import mse_loss
from utils import load_class_by_name, get_config, pop_up_image
from dataset.dataloader import get_dataloader
from modules.model import OurModel


def train_epoch(net, optim, dataloader_map, config, mode='train'):
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

    for i, images in enumerate(dataloader):
        # get the host images
        images = images.to(device=device)

        # generate the secrets message
        secret = torch.randint(0, 2, (images.size(0), num_bits)).float().to(device=device)

        with torch.set_grad_enabled(mode == 'train'):

            # forward pass
            container_image = net(secret, images)

            # attack the images
            attacked_image = net.attack_image(container_image)

            # recover the secret message
            recovered_secret = net.reverse(attacked_image)

            # calculate the loss
            image_loss = mse_loss(container_image, images)
            secret_loss = mse_loss(recovered_secret, secret)

            total_loss = lambda_image_loss * image_loss + lambda_secret_loss * secret_loss

            # backward pass
            if mode == 'train':
                optim.zero_grad()
                total_loss.backward()
                optim.step()

        print(f'Batch: #{i}, Image Loss: {image_loss.item()}, Secret Loss: {secret_loss.item()}, Total Loss: {total_loss}')


def train(name, start_epoch, end_epoch, config):
    # load the config
    channels, image_height, image_width = int(config['CHANNELS']), int(config['IMAGE_HEIGHT']), int(config['IMAGE_WIDTH'])
    num_bits = config['NUM_BITS']
    device = config['DEVICE']

    # construct the modules
    text_embedding_module = load_class_by_name(config_map['TEXT_EMBEDDING_MODULE'])(num_bits, channels, image_width,
                                                                                    image_height)
    dwt = load_class_by_name(config_map['DWT_MODULE'])()
    image_embedding_module = load_class_by_name(config_map['IMAGE_EMBEDDING_MODULE'])(channels, image_height,
                                                                                      image_width)
    attack_module = load_class_by_name(config_map['ATTACK_MODULE'])()

    # construct the model
    net = OurModel(text_embedding_module, dwt, image_embedding_module, attack_module).to(device=device)
    print(net)
    for name, param in net.named_parameters():
        print(name, param.size())

    optim = torch.optim.Adam(net.parameters(), lr=float(config['LEARNING_RATE']))

    # create the dictionary for the training
    checkpoints_path = str(config['CHECKPOINTS_PATH'])
    os.makedirs(os.path.join(checkpoints_path, name), exist_ok=True)

    # get the dataloader
    dataloader_map = get_dataloader(config)

    for epoch in range(start_epoch, end_epoch):
        # if the model is saved, load the model
        expected_model_path = os.path.join(checkpoints_path, name, f'{epoch - 1}.pth')
        expected_optim_path = os.path.join(checkpoints_path, name, f'{epoch - 1}_optim.pth')
        if epoch > 0 and os.path.exists(expected_model_path) and os.path.exists(expected_optim_path):
            net.load_state_dict(torch.load(expected_model_path))
            optim.load_state_dict(torch.load(expected_optim_path))
            print(f'Load the model and the optimizer from {epoch - 1}.pth')

        # continue the training
        train_epoch(net, optim, dataloader_map, config, mode='train')

        # validate the model
        train_epoch(net, optim, dataloader_map, config, mode='val')

        # save the state dict
        save_freq = config['SAVE_FREQ']
        if epoch % save_freq == 0:
            torch.save(net.state_dict(), os.path.join(checkpoints_path, name, f'{epoch}.pth'))
            torch.save(optim.state_dict(), os.path.join(checkpoints_path, name, f'{epoch}_optim.pth'))


if __name__ == '__main__':
    config_map = get_config()
    print(config_map)

    name = "test"
    start_epoch = 1
    end_epoch = 100

    train(name, start_epoch, end_epoch, config_map)
