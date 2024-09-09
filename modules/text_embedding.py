import torch
import torch.nn as nn


class TextEmbeddingModule(nn.Module):
    """
    Base class for text embedding modules.
    Input: 1D tensor with shape (n_bits,)
    Output: 3D tensor with shape (channels, width, height)
    """
    def __init__(self, n_bits, channels, width, height):
        super().__init__()
        self.n_bits = n_bits
        self.channels = channels
        self.width = width
        self.height = height

    def forward(self, bits, rev=False):
        raise NotImplementedError("This method should be implemented by subclasses.")


class RandomTextEmbedding(TextEmbeddingModule):
    def __init__(self, n_bits, channels, width, height):
        super().__init__(n_bits, channels, width, height)

    def forward(self, x, rev=False):
        if not rev:
            return self.transform(x)
        else:
            return self.reverse(x)

    def transform(self, bits):
        assert bits.numel() == self.n_bits, "The number of bits does not match the expected n_bits."
        return torch.rand(self.channels, self.width, self.height)

    def reverse(self, x):
        # return a random seq of 0, 1
        return torch.randint(0, 2, (self.n_bits,))


