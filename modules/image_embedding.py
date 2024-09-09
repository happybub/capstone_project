import torch
import torch.nn as nn


class ImageEmbeddingModule(nn.Module):
    """
    Base class for image embedding modules.
    Forward:
    Input: 2 4D tensors with shape (batch_size, channels, height, width)
    Output: 2 4D tensors with shape (batch_size, channels, height, width)
    Reverse:
    Input: 2 4D tensors with shape (batch_size, channels, height, width)
    Output: 2 4D tensors with shape (batch_size, channels, height, width)
    """

    def __init__(self, channels, width, height):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height

    def forward(self, image1, image2, rev=False):
        raise NotImplementedError("This method should be implemented by subclasses.")


# A example implementation of the ImageEmbeddingModule
class WeightedImageEmbedding(ImageEmbeddingModule):

    def __init__(self, channels, width, height):
        super().__init__(channels, width, height)
        self.weights = nn.Parameter(torch.Tensor(2))
        nn.init.normal_(self.weights, mean=0.5, std=0.1)

    def forward(self, image1, image2, rev=False):
        assert image1.size() == image2.size(), "Input images must have the same size"
        if not rev:
            return self.transform(image1, image2)
        else:
            return self.reverse(image1, image2)


    def transform(self, image1, image2):
        weights = torch.softmax(self.weights, dim=0)

        merged_image1 = weights[0] * image1 + weights[1] * image2
        merged_image2 = weights[1] * image1 + weights[0] * image2

        return merged_image1, merged_image2

    def reverse(self, image1, image2):
        # reverse the transform, input the merged images, output the original images
        weights = torch.softmax(self.weights, dim=0)

        original_image1 = (image1 - weights[1] * image2) / weights[0]
        original_image2 = (image2 - weights[0] * image1) / weights[1]

        return original_image1, original_image2

