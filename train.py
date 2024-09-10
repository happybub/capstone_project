from utils import get_config, pop_up_image
from dataset.dataloader import get_dataloader


if __name__ == '__main__':
    config_map = get_config()
    print(config_map)
    dataloader_map = get_dataloader(config_map)
    first_training_sample = next(iter(dataloader_map['train']))
    first_validation_sample = next(iter(dataloader_map['val']))
    first_test_sample = next(iter(dataloader_map['test']))

    pop_up_image(first_training_sample)
    pop_up_image(first_validation_sample)
    pop_up_image(first_test_sample)


