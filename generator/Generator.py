import torch

from generator.ModelLoader import ModelLoader
from generator.SentenceGenerator import SentenceGenerator
from generator.transformer_arch.SentencePreprocessor import SentencePreprocessor
from generator.Vocabulary import Vocabulary
from generator.vocab_properties import PAD_IDX
from timeit import default_timer as timer


class Generator:

    def __init__(self, batch_size):
        torch.manual_seed(0)
        self.vt = Vocabulary()
        self.VOCAB_SIZE = len(self.vt.build())
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = 1
        self.sentence_preprocessor = SentencePreprocessor(self.vt)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.ENTAILMENT = 'entailment'
        self.CONTRADICTION = 'contradiction'
        self.model_loader = ModelLoader()
        self.subgenerators = {
            self.ENTAILMENT: SentenceGenerator(self.ENTAILMENT, self.sentence_preprocessor, self.loss_fn, self.vt, self.BATCH_SIZE),
            self.CONTRADICTION: SentenceGenerator(self.CONTRADICTION, self.sentence_preprocessor, self.loss_fn, self.vt, self.BATCH_SIZE)
        }

    def train_model(self):
        for epoch in range(1, self.NUM_EPOCHS + 1):
            for option in [self.ENTAILMENT, self.CONTRADICTION]:
                start_time = timer()
                train_loss = self.subgenerators[option].train_epoch()
                end_time = timer()
                val_loss = self.subgenerators[option].evaluate()
                print(f"Epoch: {epoch}, GeneratorPart: {option}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")

    def evaluate(self):
        for option in [self.ENTAILMENT, self.CONTRADICTION]:
            val_loss = self.subgenerators[option].evaluate()
            print(f"GeneratorPart: {option}, Val loss: {val_loss:.3f}")

    def load_state(self):
        for option in [self.ENTAILMENT, self.CONTRADICTION]:
            self.model_loader.load_model(self.subgenerators[option])

    def save_state(self):
        for option in [self.ENTAILMENT, self.CONTRADICTION]:
            self.model_loader.save_model(self.subgenerators[option])

    def generate(self, src: str):
        return self.subgenerators[self.ENTAILMENT].translate(src), self.subgenerators[self.CONTRADICTION].translate(src)
