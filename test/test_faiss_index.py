import unittest
import numpy as np
from vectorization.faiss_index import build_faiss_index

class TestFaissIndex(unittest.TestCase):

    def setUp(self):
        self.vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    def test_build_faiss_index(self):
        faiss_index = build_faiss_index(self.vectors)
        self.assertIsNotNone(faiss_index)

if __name__ == '__main__':
    unittest.main()
