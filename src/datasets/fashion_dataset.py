from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import logging

logger = logging.getLogger('fashion')

class FashionDataset(Dataset):
    """
    Custom dataset class that uses metadata stored in a DataFrame and returns
    the final data: (x,y) = (img, label) -
        img is a Tensor representing the image
        label is the categorical label corresponding to the articleType
    """
    def __init__(self, df, img_path, class_map, data_transforms=None):
        """
        Args:
            df (DataFrame): pandas DataFrame containing the metadata
            img_path (string): path to the folder containing images
            class_map (dict): dictionary mapping string labels to numeric categories
            data_transforms: pytorch transforms transformations
        """
        super(FashionDataset, self).__init__()
        self.image_arr = np.asarray(df['image'].values)
        self.label_arr = np.asarray(df['articleType'].values)
        self.to_tensor = transforms.ToTensor()
        self.img_path = img_path
        self.class_map = class_map
        self.data_transforms = data_transforms

    def __getitem__(self, index):
        try:
            img_name = self.image_arr[index]
            img_as_img = Image.open(os.path.join(self.img_path, img_name))
            if img_as_img.mode != 'RGB':
                img_as_img = img_as_img.convert('RGB')

            if self.data_transforms is not None:
                img_as_tensor = self.data_transforms(img_as_img)
            else:
                img_as_tensor = self.to_tensor(img_as_img)

            label = self.class_map[self.label_arr[index]]
        except Exception as e:
            logger.error("Exception while trying to fetch image {} at index {}".format(img_name, index))
            raise e

        return (img_as_tensor, label)

    def __len__(self):
        return len(self.image_arr)
