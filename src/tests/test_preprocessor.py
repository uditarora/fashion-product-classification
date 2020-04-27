import unittest
import pandas as pd
from src.datasets.preprocess_small import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        path = '/Users/udit/code/im/data/fashion-product-images-small/myntradataset'
        self.preprocessor = Preprocessor(base_path=path)
        self.preprocessor.preprocess()
    
    def test_verify_csv(self):
        try:
            self.preprocessor.clean_csv()
            styles = pd.read_csv(self.preprocessor.csv_path)
        except:
            styles = None
        finally:
            self.assertIsNotNone(styles, "unable to load csv file")
    
    def test_verify_test_train_split(self):
        num_train = len(self.preprocessor.full_train)
        num_test = len(self.preprocessor.full_test)
        total = len(self.preprocessor.styles)
        self.assertEqual(num_test+num_train, total, "Invalid train-test split")
    
    def test_verify_subsplit(self):
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
