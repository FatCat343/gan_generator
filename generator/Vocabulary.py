from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import Iterable, List

import generator.vocab_properties as vp
from generator.SnliLoader import SnliLoader


class Vocabulary(object):
    token_transform = get_tokenizer(vp.tokenizer)
    special_symbols = vp.special_symbols
    vocab_transform: Vocab

    def __init__(self):
        self.build()

    def yield_tokens(self, data_iter: Iterable) -> List[str]:
        for data_sample in data_iter:
            yield self.token_transform(data_sample[0] + data_sample[1])

    def build(self) -> Vocab:
        snli_loader = SnliLoader()
        train_iter = snli_loader.snli_train_all()
        self.vocab_transform = build_vocab_from_iterator(self.yield_tokens(train_iter),
                                                         min_freq=1,
                                                         specials=self.special_symbols,
                                                         special_first=True)
        self.vocab_transform.set_default_index(vp.UNK_IDX)
        return self.vocab_transform

    def get_vocab(self) -> Vocab:
        return self.vocab_transform

    def get_tokenizer(self):
        return self.token_transform
