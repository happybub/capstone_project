# -*- coding: utf-8 -*-
# @Time    : 2024/10/7 21:24
# @Author  : Gan Liyifan
# @File    : check_inverse.py
import torch
from torch import nn

from modules.dwt import PRIS_DWT
from modules.model import INV_block, Hinet


class DummySubnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

def check_invertibility():
    # Initialize the block
    block = INV_block(DummySubnet)

    # Create a random input tensor
    x = torch.randn(1, block.split_len1 + block.split_len2, 32, 32)

    # Forward pass
    y = block(x, rev=False)

    # Inverse pass
    x_reconstructed = block(y, rev=True)

    # Check if the input and reconstructed input are close
    if torch.allclose(x, x_reconstructed, atol=1e-6):
        print("The block is invertible.")
    else:
        print("The block is not invertible.")

def check_hinet_invertibility():
    # Initialize the model
    channels, width, height = 3, 224, 224
    model = Hinet(channels, width, height)

    # Create dummy input data
    x = torch.randn(4, 12, 112, 112)
    y = torch.randn(4, 4, 112, 112)

    # Forward pass
    x_forward, y_forward = model(x, y, rev=False)

    # Reverse pass
    x_reconstructed, y_reconstructed = model(x_forward, y_forward, rev=True)

    # Check if the original and reconstructed inputs are the same
    x_close = torch.allclose(x, x_reconstructed, atol=1e-6)
    y_close = torch.allclose(y, y_reconstructed, atol=1e-6)

    if x_close and y_close:
        print("Hinet is invertible.")
    else:
        print("Hinet is not invertible.")

def test_dwt_reversibility():
    # Create an instance of the PRIS_DWT module
    dwt_module = PRIS_DWT()

    # Define a test input tensor
    batch_size = 1
    channels = 3
    height = 8
    width = 8
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Apply the forward transformation
    transformed_tensor = dwt_module(input_tensor)

    # Apply the reverse transformation
    reconstructed_tensor = dwt_module(transformed_tensor, rev=True)

    # Check if the reconstructed tensor is close to the original input
    if torch.allclose(input_tensor, reconstructed_tensor, atol=1e-6):
        print("The dwt module is invertible.")
    else:
        print("The dwt module is not invertible.")

if __name__ == '__main__':
    # Run the invertibility check
    check_invertibility()
    check_hinet_invertibility()
    test_dwt_reversibility()
