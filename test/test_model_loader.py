import unittest
from transformers import T5ForConditionalGeneration, T5Tokenizer
from answer_generation.model_loader import load_t5_model

class TestModelLoader(unittest.TestCase):

    def test_load_t5_model(self):
        model, tokenizer = load_t5_model()
        self.assertIsInstance(model, T5ForConditionalGeneration)
        self.assertIsInstance(tokenizer, T5Tokenizer)

if __name__ == '__main__':
    unittest.main()
