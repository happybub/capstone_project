import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: N x 3 x 224 x 224
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # 112 x 112
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 56 x 56
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 28 x 28
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 14 x 14
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten the output for the dense layer
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 1),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class Discriminator_AvgPool(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator_AvgPool, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # 112 x 112
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, stride=1, padding=1),  # 112 x 112
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 56 x 56
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 28 x 28
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class ViTDiscriminator(nn.Module):
    def __init__(self, input_channels, img_size=224, patch_size=7, embed_dim=64):
        super(ViTDiscriminator, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        # Define the patch embedding layer
        self.patch_embed = nn.Conv2d(input_channels, self.embed_dim, kernel_size=self.patch_size,
                                     stride=self.patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer Encoder
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4),
            num_layers=4
        )

        # Classifier that projects the output of transformer encoder to a single scalar
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Embed patches
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        x += self.pos_embed  # Add positional encoding

        # Transformer Encoder
        x = self.transformer_enc(x)

        # Pass the output through the classifier to get a single probability value
        x = self.classifier(x)

        return x