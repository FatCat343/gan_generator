import torch

from generator import SentenceGenerator


class ModelLoader:
    def __init__(self, saves_base_path="./.saved/spacy/new/"):
        self.saves_base_path = saves_base_path

    def load_model(self, sentence_generator: SentenceGenerator):
        checkpoint = torch.load(self.saves_base_path + sentence_generator.option + ".pt")
        sentence_generator.model.load_state_dict(checkpoint['model_state_dict'])
        sentence_generator.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model(self, sentence_generator: SentenceGenerator):
        torch.save({
            'model_state_dict': sentence_generator.model.state_dict(),
            'optimizer_state_dict': sentence_generator.optimizer.state_dict()
        }, self.saves_base_path + sentence_generator.option + ".pt")
