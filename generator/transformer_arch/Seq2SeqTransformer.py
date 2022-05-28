from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer

from generator.transformer_arch.PositionalEncoding import PositionalEncoding
from generator.transformer_arch.WordEmbedding import WordEmbedding
import generator.transformer_arch.transformer_properties as tp


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=tp.EMB_SIZE,
                                       nhead=tp.NHEAD,
                                       num_encoder_layers=tp.NUM_ENCODER_LAYERS,
                                       num_decoder_layers=tp.NUM_DECODER_LAYERS,
                                       dim_feedforward=tp.FFN_HID_DIM,
                                       dropout=dropout)
        self.generator = nn.Linear(tp.EMB_SIZE, vocab_size)
        self.src_tok_emb = WordEmbedding(vocab_size, tp.EMB_SIZE)
        self.tgt_tok_emb = WordEmbedding(vocab_size, tp.EMB_SIZE)
        self.positional_encoding = PositionalEncoding(tp.EMB_SIZE, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)
