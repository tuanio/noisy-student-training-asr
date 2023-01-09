import torch
from torch import nn
import math


class ConvSubsampling(nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: https://arxiv.org/pdf/1910.12977.pdf
    Striding Subsampling:
        "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al.
    Args:
        input_dim (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers. (encoder dim)
        subsampling_factor (int): The subsampling factor which should be a power of 2
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        input_dim: int = 80,
        feat_out: int = -1,
        conv_channels: int = -1,
        subsampling_factor: int = 4,
        activation=nn.ReLU(),
    ):
        super(ConvSubsampling, self).__init__()

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))

        in_channels = 1
        layers = []

        self._padding = 1
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False

        for i in range(self._sampling_num):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                )
            )
            layers.append(activation)
            in_channels = conv_channels

        in_length = torch.tensor(input_dim, dtype=torch.float)
        out_length = calc_length(
            in_length,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def calc_length(
        self, lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1
    ):
        """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
        add_pad: float = (padding * 2) - kernel_size
        one: float = 1.0
        for i in range(repeat_num):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
            if ceil_mode:
                lengths = torch.ceil(lengths)
            else:
                lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)

    def forward(self, x, lengths):
        lengths = self.calc_length(
            lengths,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, lengths
