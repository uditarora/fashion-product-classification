import csv
import logging
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

logger = logging.getLogger('fashion')

class Preprocessor:
    """
    Handles preprocessing of the dataset:
        - Cleans the csv file
        - Splits data into test and train split
        - Subsplits the data based on top-20 classes and rest
    """
    def __init__(self, base_path="myntradataset", process=True):
        self.base_path = base_path
        self.old_csv_path = os.path.join(base_path, "styles.csv")
        self.csv_path = os.path.join(base_path, "styles_fixed.csv")
        self.img_path = os.path.join(base_path, "images")
        if process:
            self.preprocess()

    def clean_csv(self):
        """
        Fixing bad lines in csv file (due to commas in product names)
        """
        with open(self.old_csv_path) as rf, open(self.csv_path, 'w') as wf:
            csv_reader = csv.reader(rf, delimiter=',')
            csv_writer = csv.writer(wf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in csv_reader:
                if len(row) > 10:
                    save_row = row[:8]
                    save_row.append(','.join(row[9:]))
                else:
                    save_row = row
                csv_writer.writerow(save_row)

    def get_df(self):
        """
        Returns a pandas DataFrame containing processed metadata
        """
        # Read the csv and add a column for image
        styles = pd.read_csv(self.csv_path)
        styles['image'] = styles.apply(lambda row: str(row['id']) + ".jpg", axis=1)

        # Filer out rows for which images don't exist
        img_exists = styles.apply(lambda row: os.path.exists(os.path.join(self.img_path, row['image'])), axis=1)
        self.styles = styles[img_exists]
        return styles
    
    def test_train_split(self):
        """
        Place images from even years in train-split and odd years in test-split
        """
        is_train = self.styles['year']%2==0
        is_test = self.styles['year']%2!=0

        self.full_train = self.styles[is_train]
        self.full_test = self.styles[is_test]

        return self.full_train, self.full_test
    
    def subsplit(self):
        """
        Sub-split data into two parts:
            1: images belonging to top 20 classes
            2: rest of the images
        """
        # Find the top-20 classes and the rest
        total_classes = len(self.styles['articleType'].unique())
        rest_classes = total_classes - 20
        top_articleType = self.styles.groupby('articleType').size().sort_values(ascending=False).head(20).reset_index()
        rest_articleType = self.styles.groupby('articleType').size().sort_values(ascending=True).head(rest_classes).reset_index()

        # Covert the datatype of column containing articleType to categorical
        top_articleType['articleType'] = top_articleType['articleType'].astype('category')
        rest_articleType['articleType'] = rest_articleType['articleType'].astype('category')

        self.classmap_top20 = dict(zip(top_articleType['articleType'], top_articleType['articleType'].cat.codes))
        self.classmap_ft = dict(zip(rest_articleType['articleType'], rest_articleType['articleType'].cat.codes))

        self.inv_classmap_top20 = {v: k for k, v in self.classmap_top20.items()}
        self.inv_classmap_ft = {v: k for k, v in self.classmap_ft.items()}

        # Pick the rows corresponding to the top-20 classes for pretraining and corresponding testing
        filter_topArticles = self.full_train['articleType'].isin(top_articleType['articleType'])
        train_top20_data = self.full_train[filter_topArticles]

        filter_topArticles_test = self.full_test['articleType'].isin(top_articleType['articleType'])
        test_top20_data = self.full_test[filter_topArticles_test]

        # Pick the rows corresponding to the rest of the classes for fine-tuning and corresponding testing
        train_ft_data = self.full_train[~filter_topArticles]
        test_ft_data = self.full_test[~filter_topArticles_test]

        # Split training data into train and validation splits
        train_top20_data, val_top20_data = train_test_split(train_top20_data, test_size=0.2)
        train_ft_data, val_ft_data = train_test_split(train_ft_data, test_size=0.2)

        self.data_map = {'train_top20': train_top20_data,
            'val_top20': val_top20_data,
            'train_ft': train_ft_data,
            'val_ft': val_ft_data,
            'test_top20': test_top20_data,
            'test_ft': test_ft_data}

        self.data_top20_map = {
            'train': train_top20_data,
            'val': val_top20_data,
            'test': test_top20_data
        }

        self.data_ft_map = {
            'train': train_ft_data,
            'val': val_ft_data,
            'test': test_ft_data
        }

        return self.data_top20_map, self.data_ft_map

    def preprocess(self):
        """
        Perform preprocessing sequentially
        """
        logger.info("Cleaning csv")
        self.clean_csv()
        logger.info("Reading clean csv into df")
        self.get_df()
        logger.info("Splitting into test-train")
        self.test_train_split()
        logger.info("Sub-splitting based on top-20 classes")
        self.subsplit()
