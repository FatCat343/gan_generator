import torch
from torch import nn
from torch.utils.data import DataLoader

from generator.transformer_arch.MasksBuilder import MasksBuilder
from generator.transformer_arch.Seq2SeqTransformer import Seq2SeqTransformer
from generator.SnliLoader import SnliLoader
from generator.properties import DEVICE
from generator.vocab_properties import BOS_IDX, EOS_IDX


class SentenceGenerator:
    def __init__(self, option, sentence_preprocessor, loss_fn, vocab, batch_size):
        self.BATCH_SIZE = batch_size
        self.sentence_preprocessor = sentence_preprocessor
        self.option = option
        self.vocab = vocab
        self.model = Seq2SeqTransformer(len(vocab.vocab_transform))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.loss_fn = loss_fn
        self.snli = SnliLoader()
        self.mask_builder = MasksBuilder()

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.model = self.model.to(DEVICE)

    def train_epoch(self):
        self.model.train()
        losses = 0
        train_iteration = self.snli.snli_train_option(self.option)
        train_dataloader = DataLoader(train_iteration, batch_size=self.BATCH_SIZE,
                                      collate_fn=self.sentence_preprocessor.collate_fn)
        counter = 0
        for src, tgt in train_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.mask_builder.create_mask(src, tgt_input)
            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                                src_padding_mask)
            self.optimizer.zero_grad()
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            counter += 1
        return losses / counter

    def evaluate(self):
        self.model.eval()
        losses = 0

        val_iter = self.snli.snli_valid_option(self.option)
        val_dataloader = DataLoader(val_iter, batch_size=self.BATCH_SIZE,
                                    collate_fn=self.sentence_preprocessor.collate_fn)

        counter = 0
        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.mask_builder.create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                                src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
            counter += 1

        return losses / counter

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len - 1):
            memory = memory.to(DEVICE)
            tgt_mask = (self.mask_builder.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys

    def translate(self, src_sentence: str):
        self.model.eval()
        src = self.sentence_preprocessor.text_transform(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
        return " ".join(self.vocab.vocab_transform.lookup_tokens(list(tgt_tokens.cpu().numpy()))) \
            .replace("<bos>", "") \
            .replace("<eos>", "")

    def backward_model(self, grads, sentence_batch):
        src, tgt = self.sentence_preprocessor.collate_fn(sentence_batch)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        new_grads = self.match_grads(grads, tgt_input)
        new_grads = new_grads.to(DEVICE)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.mask_builder.create_mask(src, tgt_input)
        gen_ret = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        self.optimizer.zero_grad()
        gen_ret.backward(new_grads)
        self.optimizer.step()

    def match_grads(self, grads, token_batch):
        pool_size = len(self.vocab.vocab_transform)
        new_grads = torch.zeros(token_batch.size()[0], token_batch.size()[1], pool_size)
        for batch_ind in range(grads.size()[0]):
            for token_ind in range(grads.size()[1]):
                if token_ind < token_batch.size()[0]:
                    grad = grads[batch_ind, token_ind]
                    new_grad = torch.zeros(pool_size)
                    target_pos = token_batch[token_ind, batch_ind]
                    new_grad[target_pos] = grad
                    new_grads[token_ind, batch_ind] = new_grad
        return new_grads
