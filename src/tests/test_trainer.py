import unittest
import pandas as pd
import logging
from src.datasets.preprocess import Preprocessor
from src.models.fashion_classifier import *
from src.train import *
from .util import *

logging.basicConfig(level=logging.WARN)

class TestTrainer(unittest.TestCase):
    """
    Tests various parts of the training pipeline
    """
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = Preprocessor(base_path=PATH)

    def test_model_load(self):
        """
        Checks if the model was loaded properly
        """
        try:
            model = get_ft_classifier(weight=np.arange(122), mt=False)
        except:
            model = None
        finally:
            self.assertIsNotNone(model, "Unable to create model")

        self.assertTrue(isinstance(model, FashionClassifier),
            "Incorrect model object created")

        try:
            checkpoint = torch.load(CKPT_PATH,
                map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model = None
        finally:
            self.assertIsNotNone(model, "Unable to load checkpoint")
    
    def test_model_load_mt(self):
        """
        Checks if the multitask model was loaded properly
        """
        try:
            model = get_ft_classifier(weight=np.arange(122), mt=True)
        except:
            model = None
        finally:
            self.assertIsNotNone(model, "Unable to create multitask model")

        self.assertTrue(isinstance(model, FashionClassifierMT),
            "Incorrect model object created")

        try:
            checkpoint = torch.load(CKPT_PATH_MT,
                map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model = None
        finally:
            self.assertIsNotNone(model, "Unable to load multitask checkpoint")

    def test_dataset_top20(self):
        """
        Checks if the dataset and dataloaders are created properly
        for the top-20 classes subsplit
        """
        try:
            dataloaders = {}
            for phase in ['train', 'val', 'test']:
                dataset = get_dataset(self.preprocessor, phase, subsplit='top20')
                dataloaders[phase] = get_dataloader(dataset, phase, 8)
        except:
            dataloaders = None
        finally:
            self.assertIsNotNone(dataloaders,
                "Unable to create dataloaders for the top20 classes")

    def test_dataset_ft(self):
        """
        Checks if the dataset and dataloaders are created properly
        for the fine-tune subsplit
        """
        try:
            dataloaders = {}
            for phase in ['train', 'val', 'test']:
                dataset = get_dataset(self.preprocessor, phase, subsplit='ft')
                dataloaders[phase] = get_dataloader(dataset, phase, 8)
        except:
            dataloaders = None
        finally:
            self.assertIsNotNone(dataloaders,
                "Unable to create dataloaders for the fine-tune subsplit")

    def test_trainer_setup(self):
        """
        Checks if the trainer is setup properly
        """
        try:
            _, trainer_top20, _ = setup_top20(processor=self.preprocessor)
        except:
            trainer_top20 = None
        finally:
            self.assertIsNotNone(trainer_top20,
                "Unable to setup trainer for top20 classes subsplit")

        try:
            _, trainer_ft, _ = setup_ft(processor=self.preprocessor,
                model=trainer_top20.model)
        except:
            trainer_ft = None
        finally:
            self.assertIsNotNone(trainer_ft,
                "Unable to setup trainer for fine-tune subsplit")


if __name__ == '__main__':
    unittest.main()
