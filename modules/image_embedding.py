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
        # self.weights = nn.Parameter(torch.Tensor(2))
        self.weights = torch.tensor([0.9, 0.1])

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


class Hinet(ImageEmbeddingModule):

    def __init__(self, in_1=3, in_2=3):
        super(Hinet, self).__init__()

        self.inv1 = INV_block(in_1=in_1, in_2=in_2)
        self.inv2 = INV_block(in_1=in_1, in_2=in_2)
        self.inv3 = INV_block(in_1=in_1, in_2=in_2)
        self.inv4 = INV_block(in_1=in_1, in_2=in_2)
        self.inv5 = INV_block(in_1=in_1, in_2=in_2)
        self.inv6 = INV_block(in_1=in_1, in_2=in_2)
        self.inv7 = INV_block(in_1=in_1, in_2=in_2)
        self.inv8 = INV_block(in_1=in_1, in_2=in_2)

        self.inv9 = INV_block(in_1=in_1, in_2=in_2)
        self.inv10 = INV_block(in_1=in_1, in_2=in_2)
        self.inv11 = INV_block(in_1=in_1, in_2=in_2)
        self.inv12 = INV_block(in_1=in_1, in_2=in_2)
        self.inv13 = INV_block(in_1=in_1, in_2=in_2)
        self.inv14 = INV_block(in_1=in_1, in_2=in_2)
        self.inv15 = INV_block(in_1=in_1, in_2=in_2)
        self.inv16 = INV_block(in_1=in_1, in_2=in_2)

    def forward(self, x, y, rev=False):
        x = torch.cat([x, y], dim=1)
        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        # split the output
        len = x.shape[1] // 2
        x = out[:, :len, :, :]
        y = out[:, len:, :, :]
        return x, y
