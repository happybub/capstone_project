import os

from torch.utils.data import DataLoader, random_split
from dataset.dataset import StegoDataset


def get_dataset(name, config_map):
    """
    get the dataset map, which contains the train, val, and test dataset
    config_map keys include:
    - DATA_ROOT
    """
    data_root = config_map["DATA_ROOT"]
    return StegoDataset(os.path.join(data_root, name))


def get_dataloader(config_map):
    """
    get the dataloader map, which contains the train, val, and test dataloader
    config_map keys include:
    - DATA_ROOT
    - TRAIN_BATCH_SIZE
    - VAL_BATCH_SIZE
    - TEST_BATCH_SIZE
    """
    # Load the dataset
    train_dataset = get_dataset("train", config_map)
    test_dataset = get_dataset("test", config_map)

    train_len = len(train_dataset)

    train_dataset, val_dataset = random_split(train_dataset, [int(train_len * 0.8), train_len - int(train_len * 0.8)])

    # construct the train, val, and test dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config_map["TRAIN_BATCH_SIZE"], shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=config_map["VAL_BATCH_SIZE"], shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=config_map["TEST_BATCH_SIZE"], shuffle=False, num_workers=0)

    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
