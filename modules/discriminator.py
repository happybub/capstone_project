from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: N x 3 x 224 x 224
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 112 x 112
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
    def __init__(self):
        super(Discriminator_AvgPool, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 112 x 112
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
        )

    def forward(self, x):
        return self.main(x)
