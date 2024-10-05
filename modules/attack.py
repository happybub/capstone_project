import time

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import io


class AttackModule(nn.Module):
    """
    Base class for noise attack module.
    Input: 1 4D tensors with shape (batch_size, channels, height, width)
    Output: 1 4D tensors with shape (batch_size, channels, height, width)
    """

    def __init__(self):
        super().__init__()

    def forward(self, image):
        raise NotImplementedError("This method should be implemented by subclasses.")


class GaussianNoiseAttack(AttackModule):
    """
    Add Gaussian noise to the input image
    """
    def __init__(self, mean=0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image):
        noise = torch.randn_like(image) * self.std + self.mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, min=0, max=255)


class NoneAttack(AttackModule):
    """
    A dummy attack module that does nothing.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image):
        return image


class ClippingAttack(AttackModule):
    """
    A clipping attack and keep the same size, filling with black.
    """
    def __init__(self, scale_factor=0.5):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, image):
        # 将Tensor转换为PIL Image
        to_pil = T.ToPILImage()
        pil_image = to_pil(image)

        new_size = (int(pil_image.width * self.scale_factor), int(pil_image.height * self.scale_factor))
        small_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        new_width, new_height = pil_image.size
        small_width, small_height = small_image.size
        x_offset = (new_width - small_width) // 2
        y_offset = (new_height - small_height) // 2

        new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        new_image.paste(small_image, (x_offset, y_offset))

        to_tensor = T.ToTensor()
        tensor_image_centered = to_tensor(new_image)

        return tensor_image_centered


class RotationAttack(AttackModule):
    """
    A rotation attack and keep the same size, filling with black.
    """
    def __init__(self, angle=20):
        super().__init__()
        self.angle = angle

    def forward(self, image):
        # 将Tensor转换为PIL Image
        to_pil = T.ToPILImage()
        pil_image = to_pil(image)

        rotated_image = pil_image.rotate(-self.angle, expand=True)
        new_width, new_height = pil_image.size

        rotated_width, rotated_height = rotated_image.size

        left = (rotated_width - new_width) // 2
        top = (rotated_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        cropped_image = rotated_image.crop((left, top, right, bottom))
        to_tensor = T.ToTensor()
        tensor_image_cropped = to_tensor(cropped_image)

        return tensor_image_cropped


class SaltAndPepperNoiseAttack(AttackModule):
    """
    Add salt and pepper noise to the input image
    """
    def __init__(self, salt_prob=0.01, pepper_prob=0.01):
        super().__init__()
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def forward(self, image):
        c, h, w = image.shape
        zero_matrix = torch.zeros(h, w)
        total_pixels = zero_matrix.numel()
        modify_count_salt = int(total_pixels * self.salt_prob)
        modify_count_pepper = int(total_pixels * self.pepper_prob)
        total_perm = torch.randperm(total_pixels)

        indices512 = total_perm[:modify_count_pepper]
        indices_neg512 = total_perm[total_pixels - modify_count_salt:]
        zero_matrix.view(-1)[indices512] = 512
        zero_matrix.view(-1)[indices_neg512] = -512

        zero_matrix = zero_matrix.unsqueeze(0)
        image_salted = zero_matrix.repeat(3, 1, 1)

        noisy_image = image_salted + image
        result_image = torch.clamp(noisy_image, min=-255, max=255)
        return result_image


class SaltAndPepperNoiseAttackBatch(AttackModule):
    """
    Add salt and pepper noise to the input image(batch)
    """
    def __init__(self, salt_prob=0.01, pepper_prob=0.01):
        super().__init__()
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def forward(self, image):
        n, c, h, w = image.shape
        zero_matrix = torch.zeros(h, w)
        total_pixels = zero_matrix.numel()
        modify_count_salt = int(total_pixels * self.salt_prob)
        modify_count_pepper = int(total_pixels * self.pepper_prob)
        total_perm = torch.randperm(total_pixels)

        indices512 = total_perm[:modify_count_pepper]
        indices_neg512 = total_perm[total_pixels - modify_count_salt:]
        zero_matrix.view(-1)[indices512] = 512
        zero_matrix.view(-1)[indices_neg512] = -512

        zero_matrix = zero_matrix.unsqueeze(0)
        image_salted = zero_matrix.repeat(n, c, 1, 1)

        noisy_image = image_salted + image
        result_image = torch.clamp(noisy_image, min=-255, max=255)
        return result_image


class JPEGCompressionAttack(AttackModule):
    """
    Simulate JPEG compression attack by compressing the image with a low quality factor.
    """

    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality

    def forward(self, image):
        to_pil = ToPILImage()
        pil_image = to_pil(image)

        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=self.quality)
        buffered.seek(0)

        to_tensor = ToTensor()
        tensor_image = to_tensor(Image.open(buffered))
        return tensor_image


class JPEGCompressionPRISAttack(AttackModule):
    """
    Simulate JPEG compression attack by compressing the image with a low quality factor with the method used by PRIS
    """

    def __init__(self, width=224, height=224, quality=80):
        super().__init__()
        self.width = width
        self.height = height
        self.quality = quality

    def forward(self, inputs):
        """
        param:
            inputs: a tensor with shape [n, c, h, w]

        output:
            output: a JPEG compressed tensor with shape [n, c, h, w]
        """
        from DiffJPEG import DiffJPEG

        inputs = inputs * 255
        jpeg = DiffJPEG(self.height, self.width, differentiable=True)
        quality = self.quality
        jpeg.set_quality(quality)
        outputs = jpeg(inputs)

        return outputs / 255.

class MultiAttack(AttackModule):
    """
    Randomly apply different attacks to the input image based on given probabilities.
    """
    def __init__(self, gaussian_prob=0.5, salt_pepper_prob=0.5, jpeg_prob=0, quality=50, salt_prob=0.01, pepper_prob=0.01, mean=0, std=0.01):
        super().__init__()
        self.gaussian_attack = GaussianNoiseAttack(mean=mean, std=std)
        self.salt_pepper_attack = SaltAndPepperNoiseAttackBatch(salt_prob=salt_prob, pepper_prob=pepper_prob)
        self.jpeg_attack = JPEGCompressionAttack(quality=quality)
        self.gaussian_prob = gaussian_prob
        self.salt_pepper_prob = salt_pepper_prob
        self.jpeg_prob = jpeg_prob
        self.total_prob = jpeg_prob + salt_pepper_prob + gaussian_prob
        self.range2 = self.gaussian_prob + self.salt_pepper_prob
        assert (self.gaussian_prob + self.salt_pepper_prob + self.jpeg_prob) <= 1, "Probabilities must sum to 1 or less"

    def forward(self, image):
        # Randomly select an attack based on the given probabilities
        r = torch.rand(1).item()
        if r < self.gaussian_prob / self.total_prob:
            return self.gaussian_attack.forward(image)
        elif r < self.range2 / self.total_prob:
            return self.salt_pepper_attack.forward(image)
        else:
            return self.jpeg_attack.forward(image)


def attack_testing(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    attack = ClippingAttack()
    noisy_image_tensor = attack.forward(image_tensor)

    def tensor_to_pil(tensor):
        transform1 = transforms.ToPILImage()
        return transform1(tensor)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(tensor_to_pil(image_tensor))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(tensor_to_pil(noisy_image_tensor))
    axs[1].set_title('Noisy Image')
    axs[1].axis('off')

    plt.show()


def load_images_to_tensor(image_paths, transform):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        images.append(image_tensor)
    return torch.stack(images)


def attack_testing_batch(image_paths):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # attack = MultiAttack()
    attack = JPEGCompressionPRISAttack()

    image_tensors = load_images_to_tensor(image_paths, transform)
    n = image_tensors.size(0)
    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))

    time1 = time.time()
    noisy_image_tensors = attack.forward(image_tensors)
    time2 = time.time()
    print(f"Batch processing Time taken: {time2 - time1:.4f}s")

    def tensor_to_pil(tensor):
        transform1 = transforms.ToPILImage()
        return transform1(tensor)

    for idx in range(n):
        axs[idx, 0].imshow(tensor_to_pil(image_tensors[idx]))
        axs[idx, 0].set_title('Original Image')
        axs[idx, 0].axis('off')

        axs[idx, 1].imshow(tensor_to_pil(noisy_image_tensors[idx]))
        axs[idx, 1].set_title('Noisy Image')
        axs[idx, 1].axis('off')

    plt.show()


if __name__ == "__main__":
    attack_testing_batch(['..\\data\\test\\DIV2K_valid_HR\\0801.png', '..\\data\\test\\DIV2K_valid_HR\\0802.png', '..\\data\\test\\DIV2K_valid_HR\\0803.png']
)
