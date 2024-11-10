# define the model

import torch
from torch import nn
from modules.text_embedding import TextEmbeddingModule
from modules.dwt import DWTModule
from modules.image_embedding import ImageEmbeddingModule
from modules.attack import AttackModule
from training.utils import initialize_weights, pop_up_image


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

        freq_host_image = self.dwt(host_image)

        simulated_attack = self.attack(host_image, random=True).to(device=device)

        secret_image = self.text_embedding(text_bits)

        # freq_secret_image = self.dwt(secret_image)
        freq_container, discarded = self.image_embedding(self.dwt(simulated_attack), freq_host_image, secret_image)

        container_image = self.dwt(freq_container, rev=True)

        return (freq_host_image, secret_image, freq_container, discarded), container_image

    def attack_image(self, container_image):
        noised_image = self.attack(container_image)
        return noised_image

    def reverse(self, noised_image, sample):
        attacked_container = noised_image

        freq_attacked_container = self.dwt(attacked_container)

        r_freq_container, r_secret_image = self.image_embedding(freq_attacked_container, freq_attacked_container, sample, rev=True)

        r_secret = self.text_embedding(r_secret_image, rev=True)

        return (freq_attacked_container, sample, r_freq_container, r_secret_image), r_secret


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
        # be careful the dropout, it makes the process univertable.
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
    def __init__(self, in_channels, out_channels, img_size=112, patch_size=7, embed_dim=64):
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
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=1),
            num_layers=1
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


class ResidualVitBlockQKV(nn.Module):
    def __init__(self, q_channels, k_channels, v_channels, out_channels=1, img_size=112, patch_size=7, embed_dim=128, num_heads=1):
        super(ResidualVitBlockQKV, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.q_channels = q_channels
        self.k_channels = k_channels
        self.v_channels = v_channels
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed_q = nn.Conv2d(self.q_channels, self.embed_dim, kernel_size=self.patch_size,
                                       stride=self.patch_size)
        self.patch_embed_k = nn.Conv2d(self.k_channels, self.embed_dim, kernel_size=self.patch_size,
                                       stride=self.patch_size)
        self.patch_embed_v = nn.Conv2d(self.v_channels, self.embed_dim, kernel_size=self.patch_size,
                                       stride=self.patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)

        self.to_image = nn.ConvTranspose2d(embed_dim, self.q_channels, kernel_size=self.patch_size,
                                           stride=self.patch_size)

        self.conv = nn.Conv2d(q_channels, out_channels, 3, 1, 1)

        torch.nn.init.xavier_normal_(self.patch_embed_q.weight)
        torch.nn.init.xavier_normal_(self.patch_embed_k.weight)
        torch.nn.init.xavier_normal_(self.patch_embed_v.weight)

    def forward(self, query, key, value):
        q = self.patch_embed_q(query).flatten(2).transpose(1, 2)  # [B, N_q, D_q]
        k = self.patch_embed_k(key).flatten(2).transpose(1, 2)    # [B, N_k, D]
        v = self.patch_embed_v(value).flatten(2).transpose(1, 2)  # [B, N_v, D]

        q += self.pos_embed
        k += self.pos_embed
        v += self.pos_embed

        attn_output, attn_output_weights = self.multihead_attention(q, k, v)

        # Reshape back to the output channel format
        attn_output = attn_output.transpose(1, 2).unflatten(2, (self.img_size // self.patch_size, self.img_size // self.patch_size)).contiguous()
        attn_output = self.to_image(attn_output)


        return self.conv(attn_output + query)

class INV_block(nn.Module):
    def __init__(self, clamp=2.0, channels_x=12, channels_y=1):
        super().__init__()

        self.channels_x = channels_x
        self.channels_y = channels_y

        self.clamp = clamp
        # ρ
        self.r = ResidualDenseBlock_out(self.channels_x, self.channels_y)
        # η
        self.y = ResidualVitBlockQKV(self.channels_x, self.channels_x, self.channels_x, self.channels_y)
        # self.y = ResidualDenseBlock_out(self.channels_x, self.channels_y)
        # φ
        self.f = ResidualDenseBlock_out(self.channels_y, self.channels_x)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x0, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.channels_x),
                  x.narrow(1, self.channels_x, self.channels_y))

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1, x0, x0)
            # s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1, x0, x0)
            # s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


class Hinet(ImageEmbeddingModule):
    def __init__(self, channels_x, channels_y, width, height):
        super(Hinet, self).__init__(channels_x=channels_x, channels_y=channels_y, width=width, height=height)
        self.inv_blocks = nn.ModuleList([INV_block(channels_x=self.channels_x, channels_y=self.channels_y) for _ in range(16)])
        self.pop_up_process = False

    def forward(self, key, x, y, rev=False):
        self.pop_up_process = self.pop_up_process and not self.training
        images = []
        x = torch.cat([x, y], dim=1)
        if not rev:
            out = x
            for i in range(16):
                if self.pop_up_process:
                    images.append(out[0].detach().cpu())
                out = self.inv_blocks[i](key, out)

        else:
            out = x
            for i in reversed(range(16)):
                if self.pop_up_process:
                    images.append(out[0].detach().cpu())
                out = self.inv_blocks[i](key, out, rev=True)

        # split the output
        x = out[:, :self.channels_x, :, :]
        y = out[:, self.channels_x:, :, :]

        if self.pop_up_process:
            pop_up = [[out.view(-1, self.channels_y, self.width, self.height)] for out in images]
            pop_up_image(pop_up)
        return x, y


if __name__ == '__main__':
    test = INV_block()

    # test whether it is reversible
    x = torch.randn(1, 13, 112, 112)
    out = test(x)
    print(out.shape)
    out = test(out, rev=True)
    print(out.shape)

    # check if the x and out are the same
    print((x - out).abs().max())
    print(torch.allclose(x, out))
    print(x)
    print(out)


