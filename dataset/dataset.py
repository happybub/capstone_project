import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class StegoDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        # store all the images path
        self.images_path = os.listdir(data_root)

        self.images_path = [img for img in self.images_path if img.endswith('.png')]
        self.images_path = self.images_path[:20]

        if len(self.images_path) == 0:
            raise Exception("No images found in the data_root")

        # transform to 3 * 224 * 224
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # read the idx-th item under the data_root
        img_path = os.path.join(self.data_root, self.images_path[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images_path)
