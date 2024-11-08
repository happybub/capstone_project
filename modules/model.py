# define the model

import torch
from torch import nn
from modules.text_embedding import TextEmbeddingModule
from modules.dwt import DWTModule
from modules.image_embedding import ImageEmbeddingModule
from modules.attack import AttackModule
from training.utils import initialize_weights


class OurModel(nn.Module):
    def __init__(self, text_embedding: TextEmbeddingModule, dwt: DWTModule, image_embedding: ImageEmbeddingModule,
                 attack: AttackModule):
        super(OurModel, self).__init__()
        self.text_embedding = text_embedding
        self.dwt = dwt
        self.image_embedding = image_embedding
        self.attack = attack

    def forward(self, text_bits, host_image):
        device = text_bits.device

        freq_host_image = self.dwt(host_image) # (B, 12, 112, 112)

        secret_image = text_bits.view(-1, 1, 112, 112) # (B, 1, 112, 112)

        # freq_secret_image = self.dwt(secret_image)

        freq_container, discarded = self.image_embedding(freq_host_image, secret_image)

        container_image = self.dwt(freq_container, rev=True)

        return (freq_host_image, secret_image, freq_container, discarded), container_image

    def attack_image(self, container_image):
        noised_image = self.attack(container_image)
        return noised_image

    def reverse(self, noised_image, sample):
        attacked_container = noised_image

        freq_attacked_container = self.dwt(attacked_container)

        r_freq_container, r_secret_image = self.image_embedding(freq_attacked_container, sample, rev=True)

        return freq_attacked_container, sample, r_freq_container, r_secret_image


class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class VitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=112, patch_size=7, embed_dim=64):
        super(VitBlock, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        # Define the patch embedding layer
        self.patch_embed = nn.Conv2d(in_channels, self.embed_dim, kernel_size=self.patch_size,
                                     stride=self.patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer Encoder
        # be careful the blockout, it makes the process univertable.
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4),
            num_layers=4
        )

        # Project back to the specified output channel space
        self.to_image = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=self.patch_size,
                                           stride=self.patch_size)

    def forward(self, x):
        # Embed patches
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        x += self.pos_embed  # Add positional encoding

        x = self.transformer_enc(x)

        # Reshape back to the output channel format
        x = x.transpose(1, 2).unflatten(2, (
        self.img_size // self.patch_size, self.img_size // self.patch_size)).contiguous()
        x = self.to_image(x)

        return x

class ResidualVitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=112, patch_size=7, embed_dim=256):
        super(ResidualVitBlock, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        # Define the patch embedding layer
        self.patch_embed = nn.Conv2d(in_channels, self.embed_dim, kernel_size=self.patch_size,
                                     stride=self.patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer Encoder
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=0),
            num_layers=4
        )

        # Project back to the specified output channel space
        self.to_image = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=self.patch_size,
                                           stride=self.patch_size)

    def forward(self, t):
        x = t
        # Embed patches
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        x += self.pos_embed  # Add positional encoding

        x = self.transformer_enc(x)

        # Reshape back to the output channel format
        x = x.transpose(1, 2).unflatten(2, (
        self.img_size // self.patch_size, self.img_size // self.patch_size)).contiguous()
        x = self.to_image(x)

        return self.conv(x + t)

class INV_block(nn.Module):
    def __init__(self, subnet_constructor=None, clamp=2.0, in_1=3, in_2=3):
        super().__init__()

        self.split_len1 = 12
        self.split_len2 = 1

        self.clamp = clamp
        # ρ
        self.r = ResidualDenseBlock_out(self.split_len1, self.split_len2)
        # η
        self.y = ResidualDenseBlock_out(self.split_len1, self.split_len2)
        # φ
        self.f = ResidualDenseBlock_out(self.split_len2, self.split_len1)

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
        len = 12
        x = out[:, :len, :, :]
        y = out[:, len:, :, :]
        return x, y


if __name__ == '__main__':
    test = INV_block()

    # test whether it is iverable
    x = torch.randn(1, 24, 112, 112)
    out = test(x)
    print(out.shape)
    out = test(out, rev=True)
    print(out.shape)

    # check if the x and out are the same
    print((x - out).abs().max())
    print(torch.allclose(x, out))
    print(x)
    print(out)


