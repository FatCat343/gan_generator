from generator.Generator import Generator

generator = Generator(32)
generator.train_model()
generator.save_state()
print(generator.generate("Two dogs are hiding behind the tree."))

