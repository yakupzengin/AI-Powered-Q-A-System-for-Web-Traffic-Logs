import unittest
from data_preparation.file_reader import read_log_file

class TestFileReader(unittest.TestCase):

    def test_read_log_file(self):
        log_data = read_log_file('../data/web_log_data.log')
        self.assertGreater(len(log_data), 0)
        self.assertIsInstance(log_data, list)

if __name__ == '__main__':
    unittest.main()
