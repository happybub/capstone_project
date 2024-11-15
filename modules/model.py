# define the model

import torch
from torch import nn
from modules.text_embedding import TextEmbeddingModule
from modules.dwt import DWTModule
from modules.image_embedding import ImageEmbeddingModule
from modules.attack import AttackModule
from training.utils import initialize_weights, pop_up_image


class Stego(nn.Module):
    def __init__(self, text_embedding: TextEmbeddingModule, dwt: DWTModule, image_embedding: ImageEmbeddingModule,
                 attack: AttackModule):
        super(Stego, self).__init__()
        self.text_embedding = text_embedding
        self.dwt = dwt
        self.image_embedding = image_embedding
        self.attack = attack

    def forward(self, text_bits, host_image):
        device = text_bits.device

        freq_host_image = self.dwt(host_image)

        # simulated_attack = self.attack(host_image, random=True).to(device=device)

        secret_image = self.text_embedding(text_bits)

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


class MultiLayerResidualVitBlock(nn.Module):
    def __init__(self, q_channels, k_channels, v_channels, out_channels=1, img_size=112, patch_size=7, embed_dim=64, num_heads=1):
        super(MultiLayerResidualVitBlock, self).__init__()
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

import torch
import torch.nn as nn

class SAVit(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=112, patch_size=7, embed_dim=64, num_heads=8, num_layers=2):
        super(SAVit, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed_q = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embed_k = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embed_v = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.multihead_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.to_image = nn.ConvTranspose2d(embed_dim, self.out_channels, kernel_size=self.patch_size, stride=self.patch_size)


        torch.nn.init.xavier_normal_(self.patch_embed_q.weight)
        torch.nn.init.xavier_normal_(self.patch_embed_k.weight)
        torch.nn.init.xavier_normal_(self.patch_embed_v.weight)

    def forward(self, x):
        B, C, H, W = x.shape  # [B, C, H, W]
        query = key = value = x

        q = self.patch_embed_q(query).flatten(2).transpose(1, 2)  # [B, L, D]
        k = self.patch_embed_k(key).flatten(2).transpose(1, 2)  # [B, L, D]
        v = self.patch_embed_v(value).flatten(2).transpose(1, 2) # [B, L, D]

        # if self.training:
        #     mask_ratio = 0.1
        #     mask = torch.bernoulli(torch.full((B, q.size(1), 1), fill_value=mask_ratio, device=q.device)).bool()
        #     q = q.masked_fill(mask, 0)
        #     # k = k.masked_fill(mask, 0)
        #     # v = v.masked_fill(mask, 0)


        q += self.pos_embed
        k += self.pos_embed
        v += self.pos_embed

        avg_attn_weight_list = []
        # Pass through each attention layer
        for i, (norm, mha) in enumerate(zip(self.norms, self.multihead_attentions)):
            q = norm(q)
            attn_output, attn_output_weights = mha(q, k, v)  # [B, L, D], [B, L, L]
            q = attn_output + q
            avg_attn_weight_list.append(attn_output_weights)

        from training.utils import pop_up_attention_map
        # pop_up_attention_map(torch.cat(avg_attn_weight_list).mean(dim=0))

        # [B, L, D] -> [B, D, L] -> [B, D, H, W]
        attn_output = q.transpose(1, 2).unflatten(2, (self.img_size // self.patch_size, self.img_size // self.patch_size)).contiguous()

        # [B, D, H, W] -> [B, C, H, W]
        attn_output = self.to_image(attn_output)

        return attn_output


class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=112, patch_size=7, embed_dim=64, num_heads=8):
        super(ChannelSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Embed each (H, W) channel to a feature vector
        self.avg_pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.embedding = nn.Linear(patch_size * patch_size, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.to_pixels = nn.Linear(embed_dim, img_size * img_size)

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.avg_pool(x)  # [B, C, patch_size, patch_size]
        x = x.view(B, C, -1)  # [B, C, patch_size*patch_size]
        x = self.embedding(x)  # [B, C, embed_dim]

        x = self.norm(x)
        x = x.permute(1, 0, 2)
        attn_output, attn_map = self.attn(x, x, x)  # [C, B, embed_dim]

        from training.utils import pop_up_attention_map
        # pop_up_attention_map(attn_map.permute(1, 0, 2).mean(dim=1))

        x = attn_output.permute(1, 0, 2)  # to [B, C, embed_dim]

        x = self.to_pixels(x)  # [B, C, H*W]
        x = x.view(B, C, H, W)  # [B, C, H, W]
        x = self.conv(x)

        return x




class INV_block(nn.Module):
    def __init__(self, clamp=2.0, channels_x=12, channels_y=1):
        super().__init__()

        self.channels_x = channels_x
        self.channels_y = channels_y

        self.clamp = clamp
        # ρ
        # self.y = ResidualDenseBlock_out(self.channels_x, channels_y)
        # self.r = ResidualVitBlockQKV2(q_channels=channels_y, kv_channels=channels_x, query_first=False)
        self.r = ChannelSelfAttention(in_channels=self.channels_x, out_channels=self.channels_y)
        # η
        # self.y = ResidualDenseBlock_out(self.channels_x, channels_y)
        # self.y = SAVit(in_channels=self.channels_x, out_channels=self.channels_y)
        # self.y = ResidualDenseBlock_out(q_channels=channels_y, kv_channels=channels_x, query_first=False)
        self.y = ResidualDenseBlock_out(self.channels_x, self.channels_y)
        # φ
        self.f = ResidualDenseBlock_out(self.channels_y, channels_x)
        # self.f = SAVit(in_channels=self.channels_y, out_channels=self.channels_x)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, y, rev=False):
        # x1, x2 = (x.narrow(1, 0, self.channels_x),
        #           x.narrow(1, self.channels_x, self.channels_y))
        # x1_, x2_ = (x_.narrow(1, 0, self.channels_x),
        #             x_.narrow(1, self.channels_x, self.channels_y))

        if not rev:
            new_x = x + self.f(y)
            new_y = self.e(self.r(new_x)) * y + self.y(new_x)
            # new_y = y + self.y(new_x)
            return new_x, new_y
        else:
            # pre_y = y - self.y(x)
            pre_y = (y - self.y(x)) / self.e(self.r(x))
            pre_x = x - self.f(pre_y)
            return pre_x, pre_y


class Hinet(ImageEmbeddingModule):
    def __init__(self, channels_x, channels_y, width, height):
        super(Hinet, self).__init__(channels_x=channels_x, channels_y=channels_y, width=width, height=height)
        self.inv_blocks = nn.ModuleList([INV_block(channels_x=self.channels_x, channels_y=self.channels_y) for _ in range(16)])
        self.pop_up_process = False

    def forward(self, x, y, rev=False):
        self.pop_up_process = self.pop_up_process and not self.training
        images = []
        # x = torch.cat([x, y], dim=1)
        # y = x
        for i in range(len(self.inv_blocks)) if not rev else reversed(range(len(self.inv_blocks))):
            if self.pop_up_process:
                images.append([x[0].view(-1, 3, self.height, self.width).detach().cpu(), y[0].view(-1, 1, self.height, self.width).detach().cpu()])
            x, y = self.inv_blocks[i](x, y, rev=rev)

        # split the output
        # x = x[:, :self.channels_x, :, :]
        # y = y[:, self.channels_x:, :, :]

        if self.pop_up_process:
            pop_up_image(images)
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


