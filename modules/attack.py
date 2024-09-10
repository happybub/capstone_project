import torch
from torch import nn


class AttackModule(nn.Module):
    """
    Base class for noise attack module.
    Input: 1 4D tensors with shape (batch_size, channels, height, width)
    Output: 1 4D tensors with shape (batch_size, channels, height, width)
    """

    def __init__(self):
        super().__init__()

    def forward(self, image):
        raise NotImplementedError("This method should be implemented by subclasses.")


class GaussianNoiseAttack(AttackModule):
    """
    Add Gaussian noise to the input image
    """
    def __init__(self, mean=0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image):
        noise = torch.randn_like(image) * self.std + self.mean
        return image + noise


class NoneAttack(AttackModule):
    """
    A dummy attack module that does nothing.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image):
        return image