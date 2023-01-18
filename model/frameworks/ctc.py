import torch
from torch import nn, Tensor


class CTCModel(nn.Module):
    def __init__(self, conformer_model: nn.Module, text_process):
        super().__init__()
        self.model = conformer_model
        self.ctc_loss = nn.CTCLoss()
        self.text_process = text_process

    def forward(self, feat, feat_len, target, target_len):
        out, out_len = self.model(feat, feat_len)
        if target and target_len:
            loss = self.criterion(out, target, out_len, target_len)
            return out, out_len, loss
        return out, out_len

    def criterion(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ):
        log_prob = nn.functional.log_softmax(logits, dim=-1)
        return self.ctc_loss(log_prob, targets, input_lengths, target_lengths)

    def decode(encoder_output: Tensor):
        argmax = encoder_output.squeeze(0).argmax(-1)
        return self.text_process.decode(argmax)

    def recognize(inputs: Tensor, input_lengths: Tensor):
        outputs = list()

        encoder_outputs, _ = self(inputs, input_lengths)

        for encoder_output in encoder_outputs:
            predict = decode(encoder_output)
            outputs.append(predict)

        return outputs
