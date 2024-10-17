# define the model
import time

import torch
from torch import nn
from .text_embedding import RandomTextEmbedding, TextEmbeddingModule
from .dwt import PRIS_DWT, DWTModule
from .image_embedding import WeightedImageEmbedding, ImageEmbeddingModule
from .attack import GaussianNoiseAttack, AttackModule
from utils import initialize_weights


class OurModel(nn.Module):
    def __init__(self, text_embedding: TextEmbeddingModule, dwt: DWTModule, image_embedding: ImageEmbeddingModule,
                 attack: AttackModule):
        super(OurModel, self).__init__()
        self.text_embedding = text_embedding
        self.dwt = dwt
        self.image_embedding = image_embedding
        self.attack = attack


    def forward(self, text_bits, host_image, return_extracted_secret=False):
        device = text_bits.device

        freq_host_image = self.dwt(host_image)

        # place the model on the cpu for text_embedding

        secret_image = self.text_embedding(text_bits)
        secret_image = secret_image.to(device)

        freq_secret_image = self.dwt(secret_image)

        freq_container, freq_noise = self.image_embedding(freq_host_image, freq_secret_image)

        container_image = self.dwt(freq_container, rev=True)

        if return_extracted_secret:
            return container_image, freq_noise

        return container_image

    def attack_image(self, container_image):
        noised_image = self.attack(container_image)
        return noised_image

    def reverse(self, noised_image, extracted_secret=None):
        r_container = noised_image

        r_freq_container = self.dwt(r_container)

        r_freq_noise = torch.randn_like(extracted_secret)

        r_freq_host_image, r_freq_secret_image = self.image_embedding(r_freq_container, r_freq_noise, rev=True)

        r_secret_image = self.dwt(r_freq_secret_image, rev=True)

        r_text_bits = self.text_embedding(r_secret_image, rev=True)

        return r_text_bits, r_freq_noise


class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)

        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=2.0, in_1=3, in_2=3):
        super().__init__()

        self.split_len1 = in_1 * 4
        self.split_len2 = in_2 * 4

        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


class Hinet(ImageEmbeddingModule):

    def __init__(self, channels, width, height):
        super(Hinet, self).__init__(channels, width, height)
        self.channels = channels
        in_1 = self.channels
        in_2 = 1
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
        len = out.shape[1] * self.channels // (self.channels + 1)
        x = out[:, :len, :, :]
        y = out[:, len:, :, :]
        return x, y
