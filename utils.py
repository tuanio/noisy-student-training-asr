from torch import nn
import torchaudio
import math
import torch

test_subset = ["dev-clean", "dev-other", "test-clean", "test-other"]


def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
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

    def forward(self, x, lengths):
        lengths = calc_length(
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


class LogMelSpectrogram(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, **kwargs
        )

    def forward(self, inputs):
        return self.mel_spec(inputs)


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data


def count_params(model):
    if type(model) == nn.DataParallel:
        return model.module.count_params()
    return model.count_params()


def save_state_dict(model):
    if type(model) == nn.DataParallel:
        return model.module.state_dict()
    return model.state_dict()


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
