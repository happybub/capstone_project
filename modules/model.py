# define the model
import torch
from torch import nn
from text_embedding import RandomTextEmbedding
from dwt import PRIS_DWT
from image_embedding import WeightedImageEmbedding
from attack import GaussianNoiseAttack


# TextEmbeddingModule -> DWTModule -> ImageEmbeddingModule -> AttackModule
# Reverse: AttackModule -> ImageEmbeddingModule -> DWTModule -> TextEmbeddingModule
class Stego(nn.Module):
    def __init__(self, n_bits, channels, width, height):
        super(Stego, self).__init__()
        self.text_embedding = RandomTextEmbedding(n_bits, channels, width, height)
        self.dwt = PRIS_DWT()
        self.image_embedding = WeightedImageEmbedding(channels, width, height)
        self.attack = GaussianNoiseAttack()

    def forward(self, text_bits, host_image):
        freq_host_image = self.dwt(host_image)

        secret_image = self.text_embedding(text_bits)
        secret_image = secret_image.unsqueeze(0)  # add the batch dimension
        freq_secret_image = self.dwt(secret_image)

        freq_container, freq_noise = self.image_embedding(freq_host_image, freq_secret_image)

        # discard freq_noise
        container_image = self.dwt(freq_container, rev=True)

        noised_image = self.attack(container_image)

        r_container = noised_image
        r_freq_container = self.dwt(r_container)

        # draw the r_freq_noise from Gaussian distribution
        r_freq_noise = torch.randn_like(r_freq_container)

        r_freq_host_image, r_freq_secret_image = self.image_embedding(r_freq_container, r_freq_noise, rev=True)

        # discard the host image
        r_secret_image = self.dwt(r_freq_secret_image, rev=True)

        r_text_bits = self.text_embedding(r_secret_image, rev=True)

        return r_text_bits


if __name__ == "__main__":
    # test the model
    model = Stego(10, 3, 224, 224)

    # read an sample host image
    from PIL import Image
    from torchvision import transforms

    host_image_file = Image.open("../data/train/host.jpg")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # (3, 224, 224)
    host_image = transform(host_image_file).unsqueeze(0)
    secret_text = torch.randint(0, 2, (10,))

    predicted_text = model(secret_text, host_image)
    print(predicted_text)

# pop up the image using PIL from Image
