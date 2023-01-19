from dataloader.dataset import AudioDataset

from torch.utils.data import DataLoader
from functools import partial


def collate_fn(batch, audio_transform, text_process):
    transform_fn = lambda x: audio_transform(x).squeeze().permute(1, 0)
    text_transform_fn = lambda x: text_process.text2int(text_process.tokenize(x))

    wavs = [i["wav"] for i in batch]
    feat = list(map(transform_fn, wavs))
    feat_len = torch.LongTensor([i.size(0) for i in feat])
    feat = pad_sequene(feat, batch_first=True)

    trans = [
        (idx, item["text"])
        for idx, text in enumerate(batch)
        if item["text"] is not None
    ]

    if len(trans) == len(batch):
        target = list(map(text_transform_fn, [i[1] for i in trans]))
        target_len = torch.LongTensor([i.size(0) for i in target])
        target = pad_sequence(target, batch_first=True)

        # for all label examples
        return feat, feat_len, target, target_len

    return feat, feat_len, trans


def create_dataloader(dataset_cfg: dict, dataloader_cfg: dict, text_process):

    audio_transform = MelSpectrogram(**dataset_cfg.mel_spectrogram_cfg)

    fused_collate_fn = partial(
        collate_fn, text_process=text_process, audio_transform=audio_transform
    )

    train_loader = DataLoader(
        AudioDataset(
            label_manifest_path=dataset_cfg.manifest_path.train.label,
            unlabel_manifest_path=dataset_cfg.manifest_path.train.unlabel,
        ),
        shuffle=True,
        collate_fn=fused_collate_fn,
        **dataloader_cfg
    )

    valid_loader = DataLoader(
        AudioDataset(
            label_manifest_path=dataset_cfg.manifest_path.train.valid,
        ),
        shuffle=False,
        collate_fn=fused_collate_fn,
        **dataloader_cfg
    )

    test_loader = DataLoader(
        AudioDataset(
            label_manifest_path=dataset_cfg.manifest_path.train.test,
        ),
        shuffle=False,
        collate_fn=fused_collate_fn,
        **dataloader_cfg
    )

    predict_loader = DataLoader(
        AudioDataset(
            label_manifest_path=dataset_cfg.manifest_path.train.predict,
        ),
        shuffle=False,
        collate_fn=fused_collate_fn,
        **dataloader_cfg
    )

    return train_loader, valid_loader, test_loader, predict_loader
