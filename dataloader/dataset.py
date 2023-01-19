import json
from torch import nn
from torch.utils.data import Dataset
from typing import List, Dict


class AudioDataset(Dataset):
    def __init__(self, label_manifest_path: str, unlabel_manifest_path: str = None):
        super().__init__()
        self.list_manifest = json.load(open(label_manifest_path, "r"))
        if unlabel_manifest_path:
            self.list_manifest += json.load(open(unlabel_manifest_path, "r"))

    def __len__(self):
        return len(self.list_manifest)

    def __getitem__(self, idx):
        """
        return:
        - for label example: wav, sr, text
        - for unlabel example: wav, sr, None
        """
        # label: { 'audio_filepath': ..., 'text': "hello there" }
        # unlabel: { 'audio_filepath': ...}
        sample = self.list_manifest[idx]
        wav, sr = torchaudio.load(sample.get("audio_filepath"))
        return dict(wav=wav, text=sample.get("text", None))
