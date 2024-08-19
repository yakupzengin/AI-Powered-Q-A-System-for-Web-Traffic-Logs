import unittest
import pandas as pd
from vectorization.vectorizer import vectorize_data

class TestVectorizer(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'LOG_CONTENT': ['log entry 1', 'log entry 2']})

    def test_vectorize_data(self):
        vectors, vectorizer = vectorize_data(self.df)
        self.assertEqual(vectors.shape[0], len(self.df))
        self.assertGreater(vectors.shape[1], 0)

if __name__ == '__main__':
    unittest.main()
