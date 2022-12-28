import torch
import json


class TextProcess:
    def __init__(self, dataset="libri", **kwargs):
        assert dataset in ["libri", "timit"]
        """label for timit"""
        self.base_vocabs = [
            "<p>",
            "<s>",
            "<e>",
        ]
        if dataset == "libri":
            vocab = list(
                json.load(open("phones_mapping.json", "r", encoding="utf-8")).keys()
            )
        else:
            vocab = list(json.load(open("timit_vocab.json", "r")))
        self.vocabs = self.base_vocabs + vocab

        self.n_class = len(self.vocabs)
        self.label_vocabs = dict(zip(self.vocabs, range(self.n_class)))

        self.sos_id = 1
        self.eos_id = 2
        self.blank_id = 0

    def tokenize(self, data):
        return data

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.label_vocabs[i] for i in s])

    def int2text(self, s: torch.Tensor) -> str:
        text = ""
        for i in s:
            if i in [self.sos_id, self.blank_id]:
                continue
            if i == self.eos_id:
                break
            text += " " + self.vocabs[i]
        return text

    def decode(self, argmax: torch.Tensor):
        """
        decode greedy with collapsed repeat
        """
        decode = []
        for i, index in enumerate(argmax):
            if index != self.blank_id:
                if i != 0 and index == argmax[i - 1]:
                    continue
                decode.append(index.item())
        return self.int2text(decode)
