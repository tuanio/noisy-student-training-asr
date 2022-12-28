import torch
from torch import nn, Tensor
from typing import Tuple, List


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        n_class: int,
        encoder_output_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        sos_id: int = 1,
        eos_id: int = 2,
        **kwargs
    ):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(n_class, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.out = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.SiLU()
        )

    def forward(
        self,
        targets: Tensor,
        encoder_outputs: Tensor = None,
        hidden_state: Tensor = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        input
            targets: batch of sequence label integer
            encoder_outputs (optional): output of encoder
                -> (batch size, seq len, output_dim)
            hidden_state (optional): hidden state of the last decoder
        """
        embedded = self.embedding(targets)
        outputs, hidden_state = self.lstm(embedded, hidden_state)
        outputs = self.out(outputs)
        return outputs, hidden_state
