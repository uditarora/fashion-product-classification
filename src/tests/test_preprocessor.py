import unittest
import pandas as pd
import logging
from src.datasets.preprocess import Preprocessor
from .util import *

logging.basicConfig(level=logging.WARN)

class TestPreprocessor(unittest.TestCase):
    """
    Tests the dataset preprocessor
    """
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = Preprocessor(base_path=PATH)

    def test_verify_csv(self):
        """
        Check if the csv file has been cleaned
        """
        try:
            styles = pd.read_csv(self.preprocessor.csv_path)
        except:
            styles = None
        finally:
            self.assertIsNotNone(styles, "unable to load csv file")
    
    def test_verify_test_train_split(self):
        """
        Check if the data has been split into train-test properly
        """
        num_train = len(self.preprocessor.full_train)
        num_test = len(self.preprocessor.full_test)
        total = len(self.preprocessor.styles)
        self.assertEqual(num_test+num_train, total, "Invalid train-test split")
    
    def test_verify_subsplit(self):
        """
        Check if the data has been subsplit as per top-20 classes properly
        """
        # Check if there is any intersection between top-20 and remaining values
        map1 = self.preprocessor.classmap_ft
        map2 = self.preprocessor.classmap_top20
        self.assertEqual(len(set(map1.keys()).intersection(set(map2.keys()))),
            0, "Invalid class division in subsplit - overlap found")

        top20_set, ft_set = set(), set()
        for phase in ['train', 'val']:
            top20_set = top20_set.union(set(self.preprocessor.data_top20_map[phase]['articleType'].unique()))
            ft_set = ft_set.union(set(self.preprocessor.data_ft_map[phase]['articleType'].unique()))

        top20 = len(top20_set)
        ft = len(ft_set)
        total = len(self.preprocessor.full_train['articleType'].unique())
        self.assertEqual(top20+ft, total,
            "Invalid class division in subsplit of train/val - missing classes")

        phase = 'test'
        top20 = len(self.preprocessor.data_top20_map[phase]['articleType'].unique())
        ft = len(self.preprocessor.data_ft_map[phase]['articleType'].unique())
        total = len(self.preprocessor.full_test['articleType'].unique())
        self.assertEqual(top20+ft, total,
            "Invalid class division in subsplit of test - missing classes")


if __name__ == '__main__':
    unittest.main()
