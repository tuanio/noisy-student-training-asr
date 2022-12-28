# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor, optim
import fairseq
import transformers
from torchsummary import summary
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    Trainer,
    Wav2Vec2ConformerForPreTraining,
)
import os
from pathlib import Path
from dataclasses import dataclass, field
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
import jiwer
import wandb
import time
from audio_augmentation import (
    SpeedPerturbation,
    AdaptiveSpecAugment,
    RandomBackgroundNoise,
)
from dataset import LibriLight
from utils import *
from text_process import TextProcess
import numpy as np
from transformers import Adafactor

num_hidden_layers = 2

wandb.init(
    project="speech_verification",
    name=f"conformer_{num_hidden_layers}_hidden_gen_1_libri_subsampling_swish",
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cuda:0"
print("Device:", device)
# device = 'cpu'

"""# Self-training"""

batch_size = 32

text_process = TextProcess()

n_fft = 1024
win_length = 400
hop_length = 200
n_mels = 80

# lr = 0.0005 * batch_size ** (1 / 2)
lr = 0.001
max_epochs = 100
log_idx = 2


class ConformerModel(nn.Module):
    def __init__(
        self,
        pretrained_conformer,
        input_dim,
        vocab_size,
        freq_masks=2,
        time_masks=0.05,
        freq_width=27,
        time_width=0.05,
    ):
        super().__init__()
        # input_dim: 80 (n_mels)
        feat_extract_dim = 512
        conv_channels = 256
        conformer_dim = 1024
        self.spec_augment = AdaptiveSpecAugment(
            freq_masks, time_masks, freq_width, time_width
        )
        self.conv_subsampling = ConvSubsampling(
            input_dim=input_dim, feat_out=feat_extract_dim, conv_channels=conv_channels
        )
        self.input_projection = nn.Sequential(
            nn.Linear(feat_extract_dim, conformer_dim), nn.Dropout(0.1)
        )
        self.conformer = pretrained_conformer
        self.output_projection = nn.Sequential(
            # nn.Linear(conformer_dim, conformer_dim),
            # nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(conformer_dim, vocab_size),
        )
        self.log_softmax = nn.LogSoftmax(-1)

    def freeze_conformer_blocks(self):
        for p in self.conformer.layers[0].parameters():
            p.requires_grad = False

    def forward(self, input_values, length, attention_mask=None):
        out, length = self.spec_augment(input_values, length)
        out, length = self.conv_subsampling(out, length)
        hidden_states = self.input_projection(out)
        out = self.conformer(hidden_states, attention_mask).last_hidden_state
        out = self.output_projection(out)
        out = self.log_softmax(out)
        return out, length

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def decode(encoder_output: Tensor) -> str:
    argmax = encoder_output.squeeze(0).argmax(-1)
    return text_process.decode(argmax)


def recognize(inputs: Tensor, input_lengths: Tensor, model: nn.Module) -> List[str]:
    outputs = list()

    encoder_outputs, _ = model(inputs, input_lengths)

    for encoder_output in encoder_outputs:
        predict = decode(encoder_output)
        outputs.append(predict)

    return outputs


def train_epoch(model, dataloader, optimizer, scheduler, criterion, epoch):
    size = len(dataloader)
    running_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        inputs, input_lengths, targets, target_lengths = batch
        inputs, input_lengths = inputs.to(device), input_lengths.to(device)
        targets, target_lengths = targets.to(device), target_lengths.to(device)

        outputs, output_lengths = model(inputs, input_lengths)

        loss = criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        if torch.isnan(loss).item() == True:
            break

        running_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()

        wandb.log({"train/epoch": epoch})
        wandb.log({"train/loss": loss.item()})
        if scheduler:
            wandb.log({"train/lr": scheduler.get_last_lr()[0]})
        
    return running_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, epoch, run_type="eval"):
    size = len(dataloader)
    start_time = time.perf_counter()
    running_loss = 0
    running_wer = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, input_lengths, targets, target_lengths = batch
            inputs, input_lengths = inputs.to(device), input_lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)

            outputs, output_lengths = model(inputs, input_lengths)

            loss = criterion(
                outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
            )

            predict_sequences = recognize(inputs, input_lengths, model)
            label_sequences = list(map(text_process.int2text, targets))
            wer = torch.Tensor(
                [
                    jiwer.wer(truth, hypot)
                    for truth, hypot in zip(label_sequences, predict_sequences)
                ]
            )
            wer = torch.mean(wer).item()
            running_loss += loss.item()
            running_wer += wer

            wandb.log({f"{run_type}/loss": loss.item()})
            wandb.log({f"{run_type}/wer": wer})

    return running_loss / size, running_wer / size


train_dataset = LibriLight(
    n_fft=n_fft,
    n_mels=n_mels,
    win_length=win_length,
    hop_length=hop_length,
    subsets=["light", 'dev-clean', 'dev-other'],
)

test_dataset = LibriLight(
    n_fft=n_fft,
    n_mels=n_mels,
    win_length=win_length,
    hop_length=hop_length,
    subsets=['test-clean'],
)

# test_dataset = {}
# for subset in test_subset:
#     test_dataset[subset] = LibriLight(
#         n_fft=n_fft,
#         n_mels=n_mels,
#         win_length=win_length,
#         hop_length=hop_length,
#         subset=subset,
#     )


def collate_fn(batch):
    """
    Take feature and input, transform and then padding it
    """

    specs = [i[0] for i in batch]
    input_lengths = torch.IntTensor([i.size(0) for i in specs])
    trans = [i[1] for i in batch]
    
    bs = len(specs)

    # batch, time, feature
    specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)

    trans = [text_process.text2int(s) for s in trans]
    target_lengths = torch.IntTensor([s.size(0) for s in trans])
    trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(dtype=torch.int)

    return specs, input_lengths, trans, target_lengths


train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=False,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)
# test_dataloader = {}
# for subset in test_subset:
#     test_dataloader[subset] = DataLoader(
#         test_dataset[subset],
#         batch_size=batch_size,
#         collate_fn=collate_fn,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=2,
#     )

# load pretrained big
wav2vec2_model = Wav2Vec2ConformerForPreTraining.from_pretrained(
    "facebook/wav2vec2-conformer-rel-pos-large"
)
wav2vec2_conformer = wav2vec2_model.wav2vec2_conformer.encoder
wav2vec2_conformer.layers = nn.ModuleList(
    [wav2vec2_conformer.layers[i] for i in range(num_hidden_layers)]
)


def count_params(model):
    if type(model) == nn.DataParallel:
        return model.module.count_params()
    return model.count_params()


vocab_size = text_process.n_class

conformer = ConformerModel(wav2vec2_conformer, input_dim=n_mels, vocab_size=vocab_size)
# if torch.cuda.device_count() > 1:
#     conformer = nn.DataParallel(conformer, device_ids=[0, 1])
conformer = conformer.to(device)
print(conformer)
print(
    summary(conformer, [(300, n_mels), (1,)], dtypes=[torch.float, torch.long]) 
)
ckpt = torch.load('pretrained/teacher_2_hidden_libri_subsampling_swish.pt')
conformer.load_state_dict(ckpt)

total_steps = len(train_dataloader) * max_epochs

print("Total steps:", total_steps)

criterion = nn.CTCLoss().to(device)
optimizer = optim.AdamW(conformer.parameters(), lr=lr, betas=(0.9, 0.9999))
# optimizer = Adafactor(
#     conformer.parameters(),
#     # lr=lr,
#     eps=(1e-30, 1e-3),
#     clip_threshold=1.0,
#     decay_rate=-0.8,
#     beta1=None,
#     weight_decay=0.0,
#     relative_step=True,
#     scale_parameter=True,
#     warmup_init=True,
# )
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, pct_start=0.3, total_steps=total_steps
)
# scheduler = None

config = {
    "learning_rate": lr,
    "max_epochs": max_epochs,
    "batch_size": batch_size,
    "n_fft": n_fft,
    "n_mels": n_mels,
    "num_hidden_layers": num_hidden_layers,
    "dataset": "libri-light",
    "no_params": count_params(conformer),
    "augmentation": f"SpecAugment",
}
wandb.config = config
early_stopping = EarlyStopping()

eval_loss, eval_wer = eval_epoch(
    conformer, test_dataloader, criterion, 0, 'val'
)

print("Eval wer:", eval_wer)

# print("[Training phase]" + "=" * 10)
# prev_eval_loss = 1000
# for epoch in range(max_epochs):
#     print("Epoch:", epoch)
#     train_loss = train_epoch(
#         conformer, train_dataloader, optimizer, scheduler, criterion, epoch
#     )
#     eval_loss, eval_wer = eval_epoch(
#         conformer, test_dataloader, criterion, epoch, "val"
#     )
#     early_stopping(prev_eval_loss, eval_loss)
#     if early_stopping.early_stop:
#         print("Stop at:", epoch)
#         break
#     prev_eval_loss = eval_loss

#     test_value = {}
#     print("[Testing phase]" + "=" * 10)
#     for subset in test_subset:
#         print("=" * 10 + subset + "=" * 10)
#         eval_loss, eval_wer = eval_epoch(
#             conformer, test_dataloader[subset], criterion, 0, subset
#         )
#         test_value[subset] = dict(loss=eval_loss, wer=eval_wer)

#     print("Final metric value:")
#     print(test_value)
    
# wandb.finish()

# torch.save(
#     dict(conformer_state_dict=conformer.state_dict(), config=config),
#     f"teacher_{num_hidden_layers}_hidden.pt",
# )
