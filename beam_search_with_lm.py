import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from text_process import TextProcess
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import jiwer
import heapq
from pyctcdecode import build_ctcdecoder

vocabs = [
    "t",
    "ʃ",
    "æ",
    "p",
    "ɚ",
    "w",
    "ɛ",
    "n",
    "i",
    "s",
    "v",
    "ə",
    "m",
    "eɪ",
    "d",
    "ʒ",
    "ɪ",
    "f",
    "oʊ",
    "ð",
    "uː",
    "k",
    "aɪ",
    "b",
    "ɡ",
    "j",
    "ʊɹ",
    "ɑ",
    "ɹ",
    "l",
    "ŋ",
    "ʌ",
    "z",
    "ʊ",
    "ɔː",
    "ɔ",
    "h",
    "ɑː",
    "o",
    "ɾ",
    "iə",
    "aʊ",
    "θ",
    "ɔɪ",
]
vocabs = [f" {i} " for i in vocabs]
vocab = ["", "<s>", "<e>"] + list(vocabs)
rev_vocab = dict(zip(vocab, range(len(vocab))))
print(vocab)
vocab_size = len(vocab)
print(vocab_size)

sos_id = 1
eos_id = 2
blank_id = 0


def int2text(outputs):
    return " ".join("".join([vocab[i] for i in outputs]).split())


def decode(argmax: torch.Tensor):
    decode_list = []
    for i, index in enumerate(argmax):
        if index != blank_id:
            if i != 0 and index == argmax[i - 1]:
                continue
            decode_list.append(index.item())
    return int2text(decode_list)


def recognize(encoder_outputs):
    outputs = list()
    for encoder_output in encoder_outputs:
        predict = decode(encoder_output.argmax(-1))
        outputs.append(predict)
    return outputs


def text2int(text):
    return torch.LongTensor([rev_vocab[i] for i in text])


class LM(nn.Module):
    def __init__(self, vocab_size=vocab_size, embed_size=80, hidden_size=320):
        super().__init__()
        bidirectional = False
        num_layers = 1
        lstm_dropout = 0.1
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        mul_fact = 1 + bool(bidirectional)
        self.out = nn.Linear(hidden_size * mul_fact, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, y, hidden_state=None):
        embed = self.embed(y)
        out, hidden_state = self.lstm(embed, hidden_state)
        out = self.softmax(self.out(out))
        return out, hidden_state


lm = LM()
lm.load_state_dict(
    torch.load("lstm_320.pt", map_location="cpu").get("model_state_dict")
)

example_data = torch.load("example.pt", map_location="cpu")

outputs = example_data.get("outputs")
length = example_data.get("output_length")
trans = example_data.get("trans")
# trans = list(map(lambda x: ' '.join(x[0]), trans))

greedy_out = recognize(outputs)

print(trans)
print(greedy_out)

# a1 = " ".join(trans)
# a2 = " ".join(greedy_out)
wer = jiwer.wer(trans, greedy_out)

print("Truth:", trans)
print("Greedy:", greedy_out)
print("WER greedy:", wer)

beam_width = 200

list_candidate = [(0, [], [])]  # (log_prob, input)


def ctc_collapse(inp):
    prob, indices, old_decoded = inp
    max_beam_out = torch.LongTensor(indices)
    decoded = decode(max_beam_out)
    lm_prob = 0
    if old_decoded < decoded:
        out = text2int(["<s>"] + decoded)
        lm_out, hidden_state = lm(out)
        start_idx = max(1, len(old_decoded))
        lm_prob = sum(lm_out[i, val].item() for i, val in enumerate(out[start_idx:-1]))
    return prob + lm_prob, indices, decoded


# for t in range(length):
#     new_candidates = []
#     heapq.heapify(new_candidates)
#     for log_prob, inp, old_decoded in list_candidate:
#         out = outputs[t]
#         topk = out.topk(vocab_size)
#         new_log_prob = topk.values.view(-1).tolist()
#         new_idx = topk.indices.view(-1).tolist()
#         for val, idx in zip(new_log_prob, new_idx):
#             item = log_prob + val, inp + [idx], old_decoded
#             heapq.heappush(new_candidates, item)
#             if len(new_candidates) > beam_width:
#                 heapq.heappop(new_candidates)

#     new_candidates_fuse_lm = list(map(ctc_collapse, list(new_candidates)))
#     heapq.heapify(new_candidates_fuse_lm)
#     list_candidate = list(new_candidates_fuse_lm)

# final_prob, final_candidate, old_decoded = list_candidate[-1]

# final_candidate = torch.LongTensor(final_candidate)

# final = " ".join(decode(final_candidate))

# print("Beam serach with LM:", final)
# print("Beam search with LM WER:", jiwer.wer(a1, final))

# lm_inp = list_candidate[0]
# int_lm_inp = text2int(lm_inp).unsqueeze(0)
# lm_out, hidden_state = lm(int_lm_inp)
# print(int_lm_inp.size(), lm_out.size())

# print(max_beam_out)
# beam_search_no_lm_out = ' '.join(decode(max_beam_out))
# print("Beam search no LM:", beam_search_no_lm_out)
# wer_no_lm = jiwer.wer(a1, beam_search_no_lm_out)

# print("Beam search no LM WER:", wer_no_lm)

# print(lm)

decoder = build_ctcdecoder(vocab, "phone-interpolate.3gram.arpa")

text = [decoder.decode(i.detach().numpy()) for i in outputs]
print(text)
wer = jiwer.wer(trans, text)
print(wer)
