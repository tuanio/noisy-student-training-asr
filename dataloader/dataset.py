import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class AudioDataset(Dataset):
    def __init__(self, manifest_path: str, audio_transform):
        super().__init__()
        self.list_manifest = json.load(open(manifest_path, "r"))
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.list_manifest)

    def __getitem__(self, idx):
        # { 'audio_filepath': ..., 'text': ... }
        return self.list_manifest[idx]
