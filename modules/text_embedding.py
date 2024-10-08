import time

import torch
import torch.nn as nn


class StringAsciiToTensor(nn.Module):
    """
    Class to convert a string to a tensor of ASCII bit representations.
    Input: A string
    Output: A 1D tensor of 0/1 values representing the ASCII codes in binary form.
    """

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits  # Total number of bits in the output tensor

    def forward(self, input_string):
        """
        Convert the input string to a tensor of ASCII bit representations.
        If the number of bits is less than n_bits, pad with spaces and zeros.
        """
        ascii_values = [format(ord(char), '08b') for char in input_string]

        flattened_bits = ''.join(ascii_values)
        assert len(flattened_bits) <= self.n_bits, "The number of characters should be not greater than {}.".format(int(self.n_bits / 8))
        # Convert the binary string to a list of 0/1 integers
        bit_list = [int(bit) for bit in flattened_bits]

        # Pad with spaces (ASCII 32) if necessary
        while self.n_bits - len(bit_list) >= 8:
            bit_list.extend([0, 0, 1, 0, 0, 0, 0, 0])

        # Pad with zeros if necessary
        if len(bit_list) < self.n_bits:
            bit_list += [0] * (self.n_bits - len(bit_list))

        # Convert the list to a tensor
        bit_tensor = torch.tensor(bit_list, dtype=torch.float32)
        return bit_tensor

    @staticmethod
    def reverse(bit_tensor):
        """
        Convert the tensor of 0/1 values back to the original string.
        """
        bit_list = [int(bit) for bit in bit_tensor.tolist()]
        characters = []
        for i in range(0, len(bit_list), 8):
            byte = bit_list[i:i + 8]
            if len(byte) < 8:
                break
            ascii_value = int(''.join(map(str, byte)), 2)
            characters.append(chr(ascii_value))
        return ''.join(characters)


class TextEmbeddingModule(nn.Module):
    """
    Base class for text embedding modules.
    Input: 1D tensor with shape (n_bits,)
    Output: 3D tensor with shape (channels, width, height)
    """

    def __init__(self, n_bits, channels, width, height):
        super().__init__()
        self.n_bits = n_bits
        self.channels = channels
        self.width = width
        self.height = height

    def forward(self, x, rev=False):
        if not rev:
            return self.transform(x)
        else:
            return self.reverse(x)

    def transform(self, bits):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reverse(self, x):
        raise NotImplementedError("This method should be implemented by subclasses.")


class RandomTextEmbedding(TextEmbeddingModule):
    def __init__(self, n_bits, channels, width, height):
        super().__init__(n_bits, channels, width, height)

    def transform(self, bits):
        assert bits.numel() == self.n_bits, "The number of bits does not match the expected n_bits."
        return torch.rand(self.channels, self.width, self.height)

    def reverse(self, x):
        # return a random seq of 0, 1
        return torch.randint(0, 2, (self.n_bits,))


class LinearTextEmbedding(TextEmbeddingModule):
    def __init__(self, n_bits, channels, width, height):
        super().__init__(n_bits, channels, width, height)

    def transform(self, bits):
        assert bits.numel() == self.n_bits, "The number of bits does not match the expected n_bits."
        result_tensor = torch.full((self.channels, self.width, self.height), 0, dtype=torch.float32)
        map_range = ((self.width * self.height) // 8) * 8
        for c in range(self.channels):
            for i in range(map_range):
                x = i // self.width
                y = i % self.width
                if abs(bits[i % bits.numel()].item()) > 0.5:
                    result_tensor[c, x, y] = 1
                else:
                    result_tensor[c, x, y] = 0
        return result_tensor

    def reverse(self, result_tensor):
        bits = torch.zeros(self.n_bits, dtype=torch.float32)
        bit_index = -1
        for c in range(self.channels):
            for h in range(self.height):
                for w in range(self.width):
                    bit_index += 1
                    bit_index = bit_index % self.n_bits
                    if result_tensor[c, h, w] > 0.5:
                        bits[bit_index] += 1
        threshold = self.width * self.height // self.n_bits // 2
        bits = (bits > threshold).float()
        return bits


class LinearTextEmbedding1(TextEmbeddingModule):
    def __init__(self, n_bits, channels, width, height):
        super().__init__(n_bits, channels, width, height)

    import torch

    def transform(self, bits):
        assert bits.numel() == self.n_bits, "The number of bits does not match the expected n_bits."
        bits = bits.view(self.channels, -1)[0, :]
        n = bits.numel()
        total_pixels = self.width * self.height
        k = total_pixels // n
        bits_repeated = bits.repeat(k)
        num_zeros_to_add = total_pixels - bits_repeated.numel()
        zeros_to_add = torch.zeros(num_zeros_to_add, dtype=bits.dtype)
        x_padded = torch.cat((bits_repeated, zeros_to_add))
        x_reshaped = x_padded.view(1, 224, 224)
        return x_reshaped

    def reverse(self, result_tensor):
        total_pixels = self.width * self.height
        k = total_pixels // self.n_bits

        # 将 result_tensor 转换为二维张量并置于第一个通道
        result_flat = result_tensor.view(self.channels, -1)
        result_1st_channel = result_flat[0, :k * self.n_bits]
        chunks = torch.chunk(result_1st_channel, chunks=k)
        stacked_chunks = torch.stack(chunks)
        sum_tensor = torch.sum(stacked_chunks, dim=0)
        threshold = k // 2
        sum_tensor = (sum_tensor > threshold).float()

        return sum_tensor


if __name__ == "__main__":
    start_time = time.time()
    n_bits_var = 223
    model = StringAsciiToTensor(n_bits_var)
    time1 = time.time()
    input_string_1 = "hello!"

    output_tensor = model.forward(input_string_1)
    time2 = time.time()
    #print("Tensor:", output_tensor)

    # text_embedding = LinearTextEmbedding1(n_bits_var, channels=1, width=224, height=224)
    text_embedding = LinearTextEmbedding(n_bits_var, channels=1, width=224, height=224)
    time3 = time.time()

    x_var = text_embedding.forward(output_tensor)
    time4 = time.time()

    bit_var = text_embedding.forward(x_var, rev=True)
    time5 = time.time()

    reconstructed_string = model.reverse(bit_var)

    print("1: {:.4f}s".format(time1 - start_time))
    print("2: {:.4f}s".format(time2 - time1))
    print("3: {:.4f}s".format(time3 - time2))
    print("4: {:.4f}s".format(time4 - time3))
    print("5: {:.4f}s".format(time5 - time4))
    print("String:", reconstructed_string)
