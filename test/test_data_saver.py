import unittest
import pandas as pd
import os
from data_preparation.data_saver import save_cleaned_data

class TestDataSaver(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'LOG_CONTENT': ['test log entry']})
        self.file_path = 'test_cleaned_data.csv'

    def test_save_cleaned_data(self):
        save_cleaned_data(self.df, self.file_path)
        self.assertTrue(os.path.exists(self.file_path))
        os.remove(self.file_path)

if __name__ == '__main__':
    unittest.main()
