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
from torch.nn.utils.rnn import pad_sequence
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
import numpy as np
from pprint import pprint
from audio_augmentation import SpecAugment, AdaptiveSpecAugment
from dataset import LibriLightLibriSpeechDataset
from utils import *
import time
from text_process import TextProcess

# lr = 0.002 * batch_size ** (1/2)
lr = 0.0005
max_epochs = 100
log_idx = 25
batch_size = 32
n_fft = 1024
win_length = 400  # 40ms
hop_length = 200  # 20ms
n_mels = 128
teacher_hiddens = 2
student_hiddens = 4
gen = 2
model_setting = "libri"
ckpt_version = 1
version = ckpt_version + 1
log_wandb = False

if log_wandb:
    wandb.init(
        project="speech_verification",
        name=f"student_{student_hiddens}_hidden_gen_{gen}_{model_setting}_version_{version}",
    )

device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""# Self-training"""


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
        max_time_masks=10,
        is_teacher: bool = True,
    ):
        super().__init__()
        # input_dim: 80 (n_mels)
        feat_extract_dim = 512
        conv_channels = 256
        conformer_dim = 1024
        self.spec_augment = AdaptiveSpecAugment(
            freq_masks, time_masks, freq_width, time_width, max_time_masks
        )
        self.conv_subsampling = ConvSubsampling(
            input_dim=input_dim, feat_out=feat_extract_dim, conv_channels=conv_channels
        )
        self.input_projection = nn.Sequential(
            nn.Linear(feat_extract_dim, conformer_dim),
            # nn.Dropout(0.1)
        )
        self.conformer = pretrained_conformer
        self.output_projection = nn.Sequential(
            nn.Linear(conformer_dim, conformer_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(conformer_dim, vocab_size),
        )
        self.log_softmax = nn.LogSoftmax(-1)
        self.is_teacher = is_teacher

    def freeze_conformer_blocks(self):
        for p in self.conformer.layers[0].parameters():
            p.requires_grad = False

    def forward(self, input_values, length, attention_mask=None):
        if not self.is_teacher:  # teacher sẽ không cần augment
            input_values, length = self.spec_augment(input_values, length)
        out, length = self.conv_subsampling(input_values, length)
        hidden_states = self.input_projection(out)
        out = self.conformer(hidden_states, attention_mask).last_hidden_state
        out = self.output_projection(out)
        out = self.log_softmax(out)
        return out, length

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def load_conformer_pretrained(no_hidden_layers):
    # load pretrained big
    wav2vec2_model = Wav2Vec2ConformerForPreTraining.from_pretrained(
        "facebook/wav2vec2-conformer-rel-pos-large"
    )
    wav2vec2_conformer = wav2vec2_model.wav2vec2_conformer.encoder
    wav2vec2_conformer.layers = nn.ModuleList(
        [wav2vec2_conformer.layers[i] for i in range(no_hidden_layers)]
    )
    return wav2vec2_conformer


text_process = TextProcess()
vocab_size = text_process.n_class

teacher_checkpoint = torch.load(
    f"pretrained/teacher_2_hidden_libri_subsampling_swish.pt"
)
teacher_ckpt = dict()
for key, val in teacher_checkpoint["conformer_state_dict"].items():
    teacher_ckpt[key.split(".", 1)[-1]] = val  # remove the [module.]...

wav2vec2_conformer_teacher = load_conformer_pretrained(teacher_hiddens)
teacher_model = ConformerModel(
    wav2vec2_conformer_teacher, input_dim=n_mels, vocab_size=vocab_size
)
teacher_model = teacher_model.to(device)
teacher_model.load_state_dict(teacher_ckpt)

wav2vec2_conformer_student = load_conformer_pretrained(student_hiddens)
student_model = ConformerModel(
    wav2vec2_conformer_student,
    input_dim=n_mels,
    vocab_size=vocab_size,
    time_width=0.05,
    max_time_masks=20,
    # is_teacher=False
    is_teacher=True,
)

if torch.cuda.device_count() > 1:
    student_model = nn.DataParallel(student_model, device_ids=[0, 1])
student_model = student_model.to(device)


def count_params(model):
    if type(model) == nn.DataParallel:
        return model.module.count_params()
    return model.count_params()


def save_state_dict(model):
    if type(model) == nn.DataParallel:
        return model.module.state_dict()
    return model.state_dict()


print(summary(student_model.module, [(300, n_mels,), (1,)]))


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
    start_time = time.perf_counter()
    for batch_idx, batch in enumerate(dataloader):
        inputs, input_lengths, trans = batch
        inputs, input_lengths = inputs.to(device), input_lengths.to(device)

        # teacher generate pseudo-label for student learning
        predicted = recognize(inputs, input_lengths, teacher_model)

        # replace the origin transcript of timit dataset
        for origin_trans, idx in trans:
            predicted[idx] = origin_trans

        for i in range(len(predicted)):
            if type(predicted[i]) == str:
                predicted[i] = predicted[i].split()

        predicted = [text_process.text2int(s) for s in predicted]

        check_zero = [(s.sum() > 0).item() for s in predicted]
        predicted = [predicted[i] for i, op in enumerate(check_zero) if op == True]
        inputs = inputs[check_zero]
        input_lengths = input_lengths[check_zero]

        optimizer.zero_grad()

        target_lengths = torch.IntTensor([s.size(0) for s in predicted]).to(device)
        targets = pad_sequence(predicted, batch_first=True).to(device, torch.int)

        outputs, output_lengths = model(inputs, input_lengths)

        loss = criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        loss.backward()

        optimizer.step()
        scheduler.step()

        if log_wandb:
            wandb.log({"train/epoch": epoch})
            wandb.log({"train/loss": loss.item()})
            wandb.log({"train/lr-AdamW": scheduler.get_last_lr()[0]})
            wandb.log({"train/step": batch_idx})

        # if batch_idx % log_idx == log_idx - 1:
        #     cost_time = time.perf_counter() - start_time
        #     print(f"[Epoch: {epoch:<2}|{batch_idx:<5}/{size:<5}] - Loss: {running_loss/log_idx:.2f} - Time: {cost_time:.2f}s")
        #     running_loss = 0


def eval_epoch(model, dataloader, criterion, epoch, run_type="eval"):
    running_loss = 0
    running_wer = 0
    size = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, input_lengths, trans = batch
            inputs, input_lengths = inputs.to(device), input_lengths.to(device)

            trans = [i[0] for i in trans]

            trans = [text_process.text2int(s) for s in trans]
            target_lengths = torch.IntTensor([s.size(0) for s in trans]).to(device)
            targets = pad_sequence(trans, batch_first=True).to(device, torch.int)

            outputs, output_lengths = model(inputs, input_lengths)

            loss = criterion(
                outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
            )

            predict_sequences = recognize(inputs, input_lengths, model)
            label_sequences = list(map(text_process.int2text, targets))
            list_wer = torch.Tensor(
                [
                    jiwer.wer(truth, hypot)
                    for truth, hypot in zip(label_sequences, predict_sequences)
                ]
            )
            wer = torch.mean(list_wer).item()

            running_loss += loss.item()
            running_wer += wer

            with open(f"{run_type}.txt", "w") as f:
                for truth, pred, wer_val in zip(
                    label_sequences, predict_sequences, list_wer
                ):
                    f.write(f"Actuall: [{truth}]\n")
                    f.write(f"Predict: [{pred}]\n")
                    f.write(f"WER: {wer_val * 100:.2f}%\n")
                    f.write("=" * 10 + "\n")

            if log_wandb:
                wandb.log({f"{run_type}/loss": loss.item()})
                wandb.log({f"{run_type}/wer": wer})

    return running_loss / len(dataloader), running_wer / len(dataloader)


train_dataset = LibriLightLibriSpeechDataset(subset="train")
test_dataset = LibriLightLibriSpeechDataset(subset="test")


def collate_fn(batch):
    """
    Take feature and input, transform and then padding it
    """
    trans = []
    for idx, item in enumerate(batch):
        if item[-1] == "labeled":
            trans.append((item[1], idx))
    specs = [i[0] for i in batch]
    input_lengths = torch.IntTensor([i.size(0) for i in specs])
    bs = len(specs)
    # batch, time, feature
    specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
    return specs, input_lengths, trans


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

total_steps = len(train_dataloader) * max_epochs

print("Total steps:", total_steps)

criterion = nn.CTCLoss().to(device)
optimizer = optim.AdamW(student_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, pct_start=0.25, total_steps=total_steps
)

print("Lr:", lr)

if ckpt_version:
    ckpt = torch.load(
        f"pretrained/student_{student_hiddens}_hidden_gen_2_{model_setting}_version_{ckpt_version}.pt"
    )
    start_epoch = ckpt.get("epoch", 0)
    optimizer.load_state_dict(ckpt.get("optimizer_state_dict"))
    scheduler.load_state_dict(ckpt.get("scheduler_state_dict"))

    print(student_model.module.load_state_dict(ckpt["conformer_state_dict"]))
    # print(student_model.load_state_dict(ckpt["conformer_state_dict"]))
else:
    start_epoch = 0

config = {
    "learning_rate": lr,
    "max_epochs": max_epochs,
    "batch_size": batch_size,
    "n_fft": n_fft,
    "n_mels": n_mels,
    "teacher_hiddens": teacher_hiddens,
    "student_hiddens": student_hiddens,
    "dataset": "light 10h + dev[other, clean] + test-clean + train-100",
    "student_params": count_params(student_model),
    "augmentation": f"SpecAugment",
}
if log_wandb:
    wandb.config = config


eval_loss, student_eval_wer = eval_epoch(
    student_model, test_dataloader, criterion, 0, "student_val"
)
eval_loss, teacher_eval_wer = eval_epoch(
    teacher_model, test_dataloader, criterion, 0, "teacher_val"
)

print("Teacher eval wer:", teacher_eval_wer)
print("Student eval wer:", student_eval_wer)


# best_wer = 10
# start_time = time.perf_counter()
# for epoch in range(start_epoch, max_epochs):
#     print(f"=" * 10 + f"[{epoch}, {time.perf_counter() - start_time:.2f}s]" + "=" * 10)
#     train_epoch(student_model, train_dataloader, optimizer, scheduler, criterion, epoch)
#     eval_loss, eval_wer = eval_epoch(student_model, test_dataloader, criterion, epoch, "val")
#     if eval_wer < best_wer:
#         print(f"Save model at epoch {epoch}, with WER: {eval_wer * 100:.2f}%")
#         torch.save(
#             dict(
#                 conformer_state_dict=save_state_dict(student_model),
#                 scheduler_state_dict=scheduler.state_dict(),
#                 optimizer_state_dict=optimizer.state_dict(),
#                 config=config,
#                 epoch=epoch
#             ),
#             f"student_{student_hiddens}_hidden_gen_{gen}_{model_setting}_version_{version}.pt",
#         )
#         best_wer = eval_wer

# wandb.finish()

# torch.save(
#     dict(
#         conformer_state_dict=save_state_dict(student_model),
#         scheduler_state_dict=scheduler.state_dict(),
#         optimizer_state_dict=optimizer.state_dict(),
#         config=config,
#         epoch=epoch,
#         step=batch_idx,
#     ),
#     f"student_{student_hiddens}_hidden_gen_{gen}_{model_setting}_version_{version}.pt",
# )
