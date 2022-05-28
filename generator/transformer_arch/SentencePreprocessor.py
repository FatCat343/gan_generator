from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from generator.Vocabulary import Vocabulary
from generator.vocab_properties import EOS_IDX, BOS_IDX, PAD_IDX


class SentencePreprocessor:
    def __init__(self, vocab):
        self.vocab: Vocabulary = vocab
        self.text_transform = self.sequential_transforms(self.vocab.get_tokenizer(),  # Tokenization
                                                         self.vocab.get_vocab(),  # Numericalization
                                                         self.tensor_transform)  # Add BOS/EOS and create tensor

    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    def collate_fn(self, batch):

        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform(src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform(tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch
