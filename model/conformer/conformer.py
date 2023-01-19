import torch
from torch import nn, Tensor
from model.augmentation import AdaptiveSpecAugment
from model.conformer.conv_subsampling import ConvSubsampling


class ConformerModel(nn.Module):
    def __init__(
        self,
        pretrained_conformer,
        input_dim: int = 80,
        vocab_size: int = 41,
        feat_extract_dim: int = 512,
        conv_channels: int = 256,
        conformer_dim: int = 1024,
        dropout_inp_proj: int = 0.1,
        dropout_outp_proj: int = 0.1,
        freq_masks=2,
        time_masks=0.05,
        freq_width=27,
        time_width=0.05,
        max_time_masks=10,
    ):
        super().__init__()
        self.augmentation = AdaptiveSpecAugment(
            freq_masks, time_masks, freq_width, time_width, max_time_masks
        )
        self.conv_subsampling = ConvSubsampling(
            input_dim=input_dim, feat_out=feat_extract_dim, conv_channels=conv_channels
        )
        self.input_projection = nn.Sequential(
            nn.Linear(feat_extract_dim, conformer_dim), nn.Dropout(dropout_inp_proj)
        )
        self.conformer = pretrained_conformer
        self.output_projection = nn.Sequential(
            nn.Linear(conformer_dim, conformer_dim),
            nn.SiLU(),
            nn.Dropout(dropout_outp_proj),
            nn.Linear(conformer_dim, vocab_size),
        )

    def freeze_conformer_blocks(self, n_block: int = 0):
        for l in range(n_block):
            for p in self.conformer.layers[l].parameters():
                p.requires_grad = False

    def forward(
        self,
        input_values: Tensor,
        length: Tensor,
        attention_mask: Tensor = None,
        predict: bool = False,
    ):
        if not predict:
            input_values, length = self.augmentation(input_values, length)

        out, length = self.conv_subsampling(input_values, length)
        hidden_states = self.input_projection(out)
        out = self.conformer(hidden_states, attention_mask).last_hidden_state
        out = self.output_projection(out)

        return out, length

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_state_dict(self, path_to_ckpt: str):
        ckpt = torch.load(path_to_ckpt)
        state = self.load_state_dict(ckpt)
        print("Conformer:", state)
