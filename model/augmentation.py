import torch
import random
import torchaudio
from torch import nn
import math
import os
import pathlib
import numpy as np


class SpeedPerturbation:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        #         speed_factor = random.choice([0.9, 1.0, 1.1])
        speed_factor = np.random.uniform(0.9, 1.1)
        if speed_factor == 1.0:  # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects
        )
        return transformed_audio


class RandomBackgroundNoise:
    def __init__(
        self,
        noise_dir: str,
        sample_rate: int,
        min_snr_db: int = 0,
        max_snr_db: int = 15,
    ):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f"Noise directory `{noise_dir}` does not exist")
        # find all WAV files including in sub-folders:
        noise = list(pathlib.Path(noise_dir).glob("noise/**/*.wav"))
        self.noise_files_list = noise
        if len(self.noise_files_list) == 0:
            raise IOError(f"No .wav file found in the noise directory `{noise_dir}`")

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ["remix", "1"],  # convert to mono
            ["rate", str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(
            random_noise_file, effects, normalize=True
        )
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset : offset + audio_length]
        elif noise_length < audio_length:
            noise = torch.cat(
                [noise, torch.zeros((noise.shape[0], audio_length - noise_length))],
                dim=-1,
            )

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise) / 2


class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, time_masks=10, freq_width=27, time_width=0.05):
        super().__init__()
        self._rng = random.Random()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.mask_value = 0

    @torch.no_grad()
    def forward(self, input_spec, length):
        sh = input_spec.shape
        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = self._rng.randint(0, sh[2] - self.freq_width)
                w = self._rng.randint(0, self.freq_width)
                input_spec[idx, :, x_left : x_left + w] = self.mask_value

            for i in range(self.time_masks):
                time_width = max(1, int(length[idx] * self.time_width))
                y_left = self._rng.randint(0, max(1, length[idx] - time_width))
                w = self._rng.randint(0, time_width)
                input_spec[idx, y_left : y_left + w, :] = self.mask_value
        return input_spec, length


class AdaptiveSpecAugment(nn.Module):
    def __init__(
        self,
        freq_masks=2,
        time_masks=0.05,
        freq_width=27,
        time_width=0.05,
        max_time_masks=10,
    ):
        super().__init__()
        self._rng = random.Random()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.max_time_masks = max_time_masks
        self.mask_value = 0

    @torch.no_grad()
    def forward(self, input_spec, length):
        sh = input_spec.shape
        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = self._rng.randint(0, sh[2] - self.freq_width)
                w = self._rng.randint(0, self.freq_width)
                input_spec[idx, :, x_left : x_left + w] = self.mask_value

            time_masks = min(self.max_time_masks, int(length[idx] * self.time_masks))
            for i in range(time_masks):
                time_width = max(1, int(length[idx] * self.time_width))
                y_left = self._rng.randint(0, max(1, length[idx] - time_width))
                w = self._rng.randint(0, time_width)
                input_spec[idx, y_left : y_left + w, :] = self.mask_value
        return input_spec, length
