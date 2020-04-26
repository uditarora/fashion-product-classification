# fashion-product-classification
Classification of fashion product images using transfer learning techniques

## Dataset
The Fashion Product Images Daatset is used here.
- Small version: https://www.kaggle.com/paramaggarwal/fashion-product-images-small
- Full version: https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1

## Model
A ResNet50 model with the last fully-connected layer replaced by a new layer corresponding to the number of classes in the dataset. We start with a pre-trained ResNet50 model, and retrain the whole network with the new data.

## Experimental Setup
First, the dataset is divided into train and test splits by using the year from the metadata. Images belonging to the even year numbers go to the train-split, and the others go to the test-split. This data is further sub-split into two parts - the first part consisting of images corresponding to the top-20 classes (by frequency), and the second part consisting of images corresponding to the rest of the classes.

We start with training the model on the first part of the data, and then fine-tune it on the second part.

## Updates
### Small dataset
- There are 5 images in the small dataset which are referred to in the metadata but don't exist in the dataset.
- One class from the top-20 is missing in the train-split, and many classes from the remaining classes are absent in either the test of the train splits. Conversion of class names to numerical labels needs to be done carefully.
- One of the images (`25480.jpg`) is loaded as grayscale by default. Had to convert that to RGB in the dataset loader.
- Initial experiments on the first part of the data suggest that the model is able to learn fairly quickly.
- Trained on both sets of data - the accuracy for fine-tune part is much lower.
- Calculated class-wise and average accuracy on test set and added to results.

## Results
See [RESULTS.md](RESULTS.md).

## TODO:
- ~~Evaluate model on test data.~~
- ~~Train on second part of data.~~
- Restructure code into python files
- Train on the bigger version of dataset.
- Play with augmentation, hyperparameters, and potentially model architecture to improve performance.
- Improve accuracy of smaller classes.

## Files
- `preprocess_small.ipynb` contains code for preprocessing the small version dataset.
- `fashion_classification_small.ipynb` contains the combined data processing and training code for the small version of the dataset.

## How to run
- Open `fashion_classification_small.ipynb` using jupyter and execute the cells.
