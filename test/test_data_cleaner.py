import unittest
import pandas as pd

from data_preparation.data_cleaner import clean_data


class TestDataCleaner(unittest.TestCase):

    def setUp(self):
        self.sample_data = ['121.196.121.184 - - [27/Dec/2037:12:00:00 +0530] "GET /usr HTTP/1.0" 404 4933 "-" "Mozilla/5.0 (Android 10; Mobile; rv:84.0) Gecko/84.0 Firefox/84.0" 2657',
'193.242.187.191 - - [27/Dec/2037:12:00:00 +0530] "PUT /usr/login HTTP/1.0" 404 5028 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A" 2151'
]

    def test_clean_data(self):
        cleaned = clean_data(self.sample_data)
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertGreater(len(cleaned), 0)  # Ensure some data was cleaned


if __name__ == '__main__':
    unittest.main()
