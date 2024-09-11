import os

import requests
from tqdm import tqdm
import zipfile

from torch.utils.data import DataLoader, random_split
from dataset import StegoDataset


def download_dataset(download_path, name):
    """
    Downloads the DIV2K dataset

    :param download_path: path to save the downloaded dataset
    :param name: type of the dataset, 'train' or 'test'
    """
    if name == "train":
        url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    elif name == "test":
        url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    else:
        raise ValueError("Invalid dataset name")

    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    content_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=content_size, unit='iB', unit_scale=True)

    with open(download_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            progress_bar.update(len(chunk))
            file.write(chunk)
    progress_bar.close()

    print("Download complete!")


def unzip_dataset(zip_path, extract_to):
    """
    Unzips a ZIP file to a specified directory.

    :param zip_path: Path to the ZIP file.
    :param extract_to: Directory to extract the ZIP file contents into.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Files extracted to {extract_to}")

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

if __name__ == "__main__":
    download_dataset("../data/train.zip", "train")
    unzip_dataset("../data/train.zip", "../data/train")
