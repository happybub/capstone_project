import torch
import torch.nn as nn


class DWTModule(nn.Module):
    """
    Base class for DWT modules.
    Forward:
    Input: 4D tensor with shape (batch_size, channels, height, width)
    Output: 4D tensor with shape (batch_size, n^2 * channels, height // n, width // n)
    Reverse:
    Input: 4D tensor with shape (batch_size, n^2 * channels, height // n, width // n)
    Output: 4D tensor with shape (batch_size, channels, height, width)
    """

    def __init__(self):
        super().__init__()

    def forward(self, image, rev=False):
        raise NotImplementedError("This method should be implemented by subclasses.")


class PRIS_DWT(DWTModule):
    """
    DWT module from PRIS.
    Forward:
    Input: 4D tensor with shape (batch_size, channels, height, width)
    Output: 4D tensor with shape (batch_size, 4 * channels, height // 2, width // 2)

    Reverse:
    Input: 4D tensor with shape (batch_size, 4 * channels, height // 2, width // 2)
    Output: 4D tensor with shape (batch_size, channels, height, width)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, rev=False):
        assert x.dim() == 4, "Input must be a 4D tensor with shape (batch_size, channels, height, width)"
        if not rev:
            return self.transform(x)
        else:
            return self.reverse(x)

    def transform(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2

        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def reverse(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

class Clipping(DWTModule):
    """
    DWT module from PRIS.
    Forward:
    Input: 4D tensor with shape (batch_size, channels, height, width)
    Output: 4D tensor with shape (batch_size, 4 * channels, height // 2, width // 2)

    Reverse:
    Input: 4D tensor with shape (batch_size, 4 * channels, height // 2, width // 2)
    Output: 4D tensor with shape (batch_size, channels, height, width)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, rev=False):
        assert x.dim() == 4, "Input must be a 4D tensor with shape (batch_size, channels, height, width)"
        if not rev:
            return self.transform(x)
        else:
            return self.reverse(x)

    def transform(self, x):
        B, C, H, W = x.shape
        H2 = H // 2
        W2 = W // 2

        x1 = x[:, :, :H2, :W2]
        x2 = x[:, :, H2:, :W2]
        x3 = x[:, :, :H2, W2:]
        x4 = x[:, :, H2:, W2:]

        return torch.cat((x1, x2, x3, x4), 1)

    def reverse(self, x):
        B, C_, H_, W_ = x.shape
        H = 2 * H_
        W = 2 * W_
        C = C_ // 4
        x_reshaped = x.view(B, 4, C, H_, W_)
        top = torch.cat([x_reshaped[:, 0, :, :, :], x_reshaped[:, 2, :, :, :]], dim=3)
        bottom = torch.cat([x_reshaped[:, 1, :, :, :], x_reshaped[:, 3, :, :, :]], dim=3)
        return torch.cat([top, bottom], dim=2)


# class PRIS_DWT(DWTModule):
#     """
#     DWT module from PRIS.
#     Forward:
#     Input: 4D tensor with shape (batch_size, channels, height, width)
#     Output: 4D tensor with shape (batch_size, 8 * channels, height // 2, width // 2)
#
#     Reverse:
#     Input: 4D tensor with shape (batch_size, 8 * channels, height // 2, width // 2)
#     Output: 4D tensor with shape (batch_size, channels, height, width)
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x, rev=False):
#         assert x.dim() == 4, "Input must be a 4D tensor with shape (batch_size, channels, height, width)"
#         if not rev:
#             return self.transform(x)
#         else:
#             return self.reverse(x)
#
#     def transform(self, x):
#         x01 = x[:, :, 0::2, :] / 2
#         x02 = x[:, :, 1::2, :] / 2
#
#         x1 = x01[:, :, :, 0::2]
#         x2 = x02[:, :, :, 0::2]
#         x3 = x01[:, :, :, 1::2]
#         x4 = x02[:, :, :, 1::2]
#
#         x_LL = x1 + x2 + x3 + x4
#         x_HL = -x1 - x2 + x3 + x4
#         x_LH = -x1 + x2 - x3 + x4
#         x_HH = x1 - x2 - x3 + x4
#
#         x_LL2 = x_LL[:, :, 0::2, :] / 2
#         x_HL2 = x_HL[:, :, 0::2, :] / 2
#         x_LH2 = x_LH[:, :, 0::2, :] / 2
#         x_HH2 = x_HH[:, :, 0::2, :] / 2
#
#         x_LL2 = x_LL2[:, :, :, 0::2]
#         x_HL2 = x_HL2[:, :, :, 0::2]
#         x_LH2 = x_LH2[:, :, :, 0::2]
#         x_HH2 = x_HH2[:, :, :, 0::2]
#
#         return torch.cat((x_LL, x_HL, x_LH, x_HH, x_LL2, x_HL2, x_LH2, x_HH2), 1)
#
#     def reverse(self, x):
#         r = 2
#         in_batch, in_channel, in_height, in_width = x.size()
#         out_batch, out_channel, out_height, out_width = in_batch, int(
#             in_channel / 8), r * in_height, r * in_width
#         x1 = x[:, 0:out_channel, :, :] / 2
#         x2 = x[:, out_channel:out_channel * 2, :, :] / 2
#         x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
#         x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
#         x5 = x[:, out_channel * 4:out_channel * 5, :, :] / 2
#         x6 = x[:, out_channel * 5:out_channel * 6, :, :] / 2
#         x7 = x[:, out_channel * 6:out_channel * 7, :, :] / 2
#         x8 = x[:, out_channel * 7:out_channel * 8, :, :] / 2
#
#         h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
#
#         h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4 + x5 - x6 - x7 + x8
#         h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8
#         h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4 + x5 + x6 - x7 - x8
#         h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
#
#         return h


if __name__ == '__main__':
    # test the module
    dwt = Clipping()
    x = torch.rand(1, 3, 256, 256)
    y = dwt(x)
    z = dwt(y, rev=True)

    # measure the difference between x and z
    assert torch.allclose(x, z, atol=1e-6), "The reverse operation is not correct."
    print(f"The {dwt.__class__.__name__} module is correct.")

