import torchaudio
from utils import LogMelSpectrogram
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from pathlib import Path
import torch
from typing import List


class LibriLight(Dataset):
    def __init__(
        self,
        data_path: str = "data/libri-phone",
        subsets: List[str] = ["light", "dev-clean", "dev-other"],
        n_fft: int = 2048,
        n_mels: int = 80,
        win_length: int = 400,
        hop_length: int = 100,
    ):
        super().__init__()
        # assert subset in ['light', 'dev-clean', 'dev-other', 'test-clean', 'test-other'], 'Not found subset'
        df = pd.read_csv(data_path + os.sep + "phones.csv")
        df = df[df.subset.isin(subsets)].drop("subset", axis=1)
        df.label = df.label.map(eval)
        df.path = data_path + os.sep + df.path
        df.path = df.path.apply(lambda x: x.replace("\\", os.sep))
        self.walker = df.to_dict("records")

        self.feature_transform = LogMelSpectrogram(
            n_fft=n_fft, n_mels=n_mels, win_length=win_length, hop_length=hop_length
        )

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]
        label = item["label"]
        wave, sr = torchaudio.load(item["path"])

        specs = self.feature_transform(wave)
        specs = specs.permute(0, 2, 1)
        specs = specs.squeeze()

        return specs, label


class LibriLightLibriSpeechDataset(Dataset):
    def __init__(
        self,
        light_data_path: str = "data/libri-phone",
        subset: str = "train",
        libri_clean_path: str = "data/LibriSpeech",
        n_fft: int = 1024,
        n_mels: int = 128,
        win_length: int = 400,
        hop_length: int = 200,
        **kwargs,
    ):
        super().__init__()
        """
        subset \in ['train', 'val', 'test']
        """
        self.list_url = []
        is_test = True
        if subset == "train":
            self.list_url = [libri_clean_path + "/train-clean-360"]
            is_test = False

        sep = os.sep
        self.libri_walker = []
        for path in self.list_url:
            files_path = f"*{sep}*{sep}*" + ".flac"
            walker = [(str(p.stem), path) for p in Path(path).glob(files_path)]
            self.libri_walker.extend(walker)

        if subset == "train":
            subsets = ["light", "dev-clean", "dev-other", "test-other"]
        else:
            subsets = ["test-clean"]

        df = pd.read_csv(light_data_path + os.sep + "phones.csv")
        df = df[df.subset.isin(subsets)].drop("subset", axis=1)
        df.label = df.label.map(eval)
        df.path = light_data_path + os.sep + df.path
        df.path = df.path.apply(lambda x: x.replace("\\", os.sep))
        self.light_walker = df.to_dict("records")

        self.walker = self.libri_walker + self.light_walker

        sample_rate = 16000
        self.feature_transform = LogMelSpectrogram(
            n_fft=n_fft, n_mels=n_mels, win_length=win_length, hop_length=hop_length
        )

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]
        if type(item) == tuple:
            return self.load_librispeech_item(item)
        return self.load_libri_light_item(item)

    def load_libri_light_item(self, item):
        label = item["label"]
        wave, sr = torchaudio.load(item["path"])

        specs = self.feature_transform(wave)
        specs = specs.permute(0, 2, 1)
        specs = specs.squeeze()

        return specs, label, "labeled"

    def load_librispeech_item(self, item):
        """
        transform audio pack to spectrogram
        """
        fileid, path = item

        speaker_id, chapter_id, utterance_id = fileid.split("-")
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + ".flac"
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_audio)

        spectrogram = self.feature_transform(waveform)
        spectrogram = spectrogram.squeeze().permute(1, 0)

        return spectrogram, "unlabel"


# class TimitLibriSpeechDataset(Dataset):
#     def __init__(
#         self,
#         timit_data_root: str = 'data/timit',
#         timit_csv_path: str = 'data/timit/timit_pronunciation.csv',
#         libri_clean_path: str = 'data/libri/LibriSpeech',
#         libri_other_path: str = '',
#         subset: str = 'train',
#         n_fft: int = 512,
#         n_mels: int = 80,
#         **kwargs,
#     ):
#         super().__init__()
#         """
#         subset \in ['train', 'val', 'test']
#         """
#         self.list_url = []
#         is_test = True
#         if subset == "train":
#             self.list_url = [libri_clean_path + "train-clean-100"]
#             is_test = False

#         sep = os.sep
#         self.libri_walker = []
#         for path in self.list_url:
#             files_path = f"*{sep}*{sep}*" + '.flac'
#             walker = [(str(p.stem), path) for p in Path(path).glob(files_path)]
#             self.libri_walker.extend(walker)

#         df = pd.read_csv(timit_csv_path, index_col=0)
#         df = df[df.is_test == is_test].drop("is_test", axis=1)
#         df.path = df.path.apply(lambda x: timit_data_root + os.sep + x)
#         df.trans = df.trans.str.split("|")
#         self.is_test = is_test

#         self.timit_walker = df.to_dict("records")

#         self.walker = self.libri_walker + self.timit_walker

#         sample_rate = 16000
#         self.feature_transform = LogMelSpectrogram(n_fft=n_fft, n_mels=n_mels)
#         self.augmentation = ComposeTransform([
#             SpeedPerturbation(sample_rate),
#             RandomBackgroundNoise(sample_rate, max_snr_db=30)
#         ])
#         self.augment_prob = 0.80


#     def __len__(self):
#         return len(self.walker)

#     def __getitem__(self, idx):
#         item = self.walker[idx]
#         if type(item) == tuple:
#             return self.load_librispeech_item(item)
#         return self.load_timit_item(item)

#     def load_timit_item(self, item):
#         trans = item["trans"]
#         wave, sr = torchaudio.load(item["path"])

#         is_augment = np.random.choice(2, p=(1 - self.augment_prob, self.augment_prob))
#         if is_augment and not self.is_test:
#             wave = self.augmentation(wave)

#         specs = self.feature_transform(wave)
#         specs = specs.permute(0, 2, 1)
#         specs = specs.squeeze()
#         return specs, trans, 'labelled'

#     def load_librispeech_item(self, item):
#         """
#         transform audio pack to spectrogram
#         """
#         fileid, path = item

#         speaker_id, chapter_id, utterance_id = fileid.split("-")
#         fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
#         file_audio = fileid_audio + '.flac'
#         file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

#         # Load audio
#         waveform, sample_rate = torchaudio.load(file_audio)

#         is_augment = np.random.choice(2, p=(1 - self.augment_prob, self.augment_prob))
#         if is_augment:
#             waveform = self.augmentation(waveform)

#         spectrogram = self.feature_transform(waveform)
#         spectrogram = spectrogram.squeeze().permute(1, 0)

#         return spectrogram, 'unlabel'


class TimitLibriSpeechDataset(Dataset):
    def __init__(
        self,
        timit_data_root: str = "data/timit",
        timit_csv_path: str = "data/timit/timit_pronunciation.csv",
        libri_clean_path: str = "data/libri/LibriSpeech",
        libri_other_path: str = "",
        subset: str = "train",
        n_fft: int = 512,
        n_mels: int = 80,
        **kwargs,
    ):
        super().__init__()
        """
        subset \in ['train', 'val', 'test']
        """
        self.list_url = []
        is_test = True
        if subset == "train":
            self.list_url = [libri_clean_path + "train-clean-100"]
            is_test = False

        sep = os.sep
        self.libri_walker = []
        for path in self.list_url:
            files_path = f"*{sep}*{sep}*" + ".flac"
            walker = [(str(p.stem), path) for p in Path(path).glob(files_path)]
            self.libri_walker.extend(walker)

        df = pd.read_csv(timit_csv_path, index_col=0)
        df = df[df.is_test == is_test].drop("is_test", axis=1)
        df.path = df.path.apply(lambda x: timit_data_root + os.sep + x)
        df.trans = df.trans.str.split("|")
        self.is_test = is_test

        self.timit_walker = df.to_dict("records")

        self.walker = self.libri_walker + self.timit_walker

        sample_rate = 16000
        self.feature_transform = LogMelSpectrogram(n_fft=n_fft, n_mels=n_mels)

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]
        if type(item) == tuple:
            return self.load_librispeech_item(item)
        return self.load_timit_item(item)

    def load_timit_item(self, item):
        trans = item["trans"]
        wave, sr = torchaudio.load(item["path"])

        specs = self.feature_transform(wave)
        specs = specs.permute(0, 2, 1)
        specs = specs.squeeze()
        return specs, trans, "labelled"

    def load_librispeech_item(self, item):
        """
        transform audio pack to spectrogram
        """
        fileid, path = item

        speaker_id, chapter_id, utterance_id = fileid.split("-")
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + ".flac"
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_audio)

        spectrogram = self.feature_transform(waveform)
        spectrogram = spectrogram.squeeze().permute(1, 0)

        return spectrogram, "unlabel"


class TimitDataset(Dataset):
    def __init__(
        self,
        data_root: str = "data/timit",
        csv_path: str = "data/timit/timit_pronunciation.csv",
        n_fft: int = 159,
        n_mels: int = 80,
        is_test: bool = False,
        **kwargs,
    ):
        super().__init__()
        df = pd.read_csv(csv_path, index_col=0)
        df = df[df.is_test == is_test].drop("is_test", axis=1)
        df.path = df.path.apply(lambda x: data_root + os.sep + x)
        df.trans = df.trans.str.split("|")
        self.is_test = is_test
        sample_rate = 16000

        self.walker = df.to_dict("records")
        self.feature_transform = LogMelSpectrogram(n_fft=n_fft, n_mels=n_mels)

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]
        trans = item["trans"]
        wave, sr = torchaudio.load(item["path"])

        specs = self.feature_transform(wave)

        specs = specs.permute(0, 2, 1)
        specs = specs.squeeze()

        return specs, trans
