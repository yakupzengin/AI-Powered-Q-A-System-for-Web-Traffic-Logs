import unittest
import pandas as pd
from answer_generation.query_processor import QueryProcessor
from vectorization.vectorizer import vectorize_data
from vectorization.faiss_index import build_faiss_index

class TestQueryProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = QueryProcessor()
        self.df = pd.DataFrame({'LOG_CONTENT': ['test log entry'], 'URL': ['test_page']})
        self.vectors, self.vectorizer = vectorize_data(self.df)
        self.faiss_index = build_faiss_index(self.vectors)

    def test_generate_answer(self):
        answer = self.processor.generate_answer(self.df, self.faiss_index, 'test query', self.vectorizer)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

if __name__ == '__main__':
    unittest.main()
