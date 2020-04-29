# fashion-product-classification
Classification of fashion product images using transfer learning techniques

## How to run
Install the requirements in `requirements.txt` and run:
```bash
$ python main.py --data <path_to_dataset> --ckpt <path_to_checkpoint_folder>
```
#### Tests
Update path in `src/tests/util.py` and run `python -m unittest discover` from the root directory.

## Files
### Source code
- `src/datasets/` contains code for preprocessing and loading the dataset
- `src/models` contains the code for creating and fetching models
- `src/tests/` cotains testing code
- `train.py` contians the Trainer class and test training code
### Experiments
- `fashion_classification_small.ipynb` contains code used to obtain the latest results on small dataset.
- `visualize.ipynb` contains code to visualize the results of the model.
- `experiments/metadata_experiments.ipynb` contains code used to evaluate metadata.
- `experiments/preprocess_small.ipynb` contains code for preprocessing the small version dataset.
- `experiments/fashion_classification_small.ipynb` contains the combined data processing and training code for the small version of the dataset.

## Dataset
The Fashion Product Images Daatset is used in this repo.
- Small version: https://www.kaggle.com/paramaggarwal/fashion-product-images-small
- Full version: https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1

## Experimental Setup
### Model
A ResNet50 model with the last fully-connected layer replaced by a new layer corresponding to the number of classes in the dataset. We start with a pre-trained ResNet50 model, and retrain the whole network with the new data.

First, the dataset is divided into train and test splits by using the year from the metadata. Images belonging to the even year numbers go to the train-split, and the others go to the test-split. This data is further sub-split into two parts - the first part consisting of images corresponding to the top-20 classes (by frequency), and the second part consisting of images corresponding to the rest of the classes.

We start with training the model on the first part of the data, and then fine-tune it on the second part.

## Observations
- SGD with momentum optimizer seems to learn faster than Adam optimizer and achieved a better validation accuracy in just 15 epochs than Adam did in over 50 epochs on the top-20 classes. Hence SGD was used for all further experiments.
- The accuracy of most classes for which there was no training data in the subsplit is 0.
- The images in the dataset seem to have the product in the center and there is low amount of empty space. Based on this observation, the data augmentaion strategy was changed to using a crop scale between (0.5, 1) while applying data augmenation. On the small dataset - This improved the average Top-1 test accuracy by **4.52%** and average Top-5 accuracy by **0.35%** for the top-20 classes. For the fine-tune subsplit, the accuracies went up by **0.79%** and **0.05%**.
- Training a simple Naive Bayes classifier on the product display name achieves test accuracy of 85.23% on top-20 subsplit and 42.55% on the fine-tune subsplit. This performance is surprisingly close to the performance of the CNN trained on the images.
- Introducing shuffling in the training data-loaders improved the Top-1 accuracy by **1.13%** and Top-5 accuracy by **0.49%** for the fine-tune subsplit.

## Updates
### Small dataset
- There are 5 images in the small dataset which are referred to in the metadata but don't exist in the dataset.
- One class from the top-20 is missing in the train-split, and many classes from the remaining classes are absent in either the test of the train splits. Conversion of class names to numerical labels needs to be done carefully.
- One of the images (`25480.jpg`) is loaded as grayscale by default. Had to convert that to RGB in the dataset loader.
- Initial experiments on the first part of the data suggest that the model is able to learn fairly quickly.
- Trained on both sets of data - the accuracy for fine-tune part is much lower.
- Calculated class-wise and average accuracy on test set and added to results.
- Made the code modular by restructuring into classes.
- Added unit tests for data processing.
- Improved test accuracy by updating data augmentation and choice of optimizer.
- Tried re-training the fine-tune model on the bottom 50 classes with fewer samples by applying more data augmentations. It improved the test accuracy of some classes but led to a decrease in the average test accuracy, possibly due to overfitting on the smaller classes.
- Improved accuracy by shuffling data during training.

## Results
See [RESULTS.md](RESULTS.md).

## TODO:
- ~~Evaluate model on test data.~~
- ~~Train on second part of data.~~
- ~~Restructure code into python files.~~
- Train on the bigger version of dataset.
- ~~Play with augmentation, hyperparameters, and potentially model architecture to improve performance.~~
- Improve accuracy of smaller classes.

## Ideas:
- Apply offline data augmentation on the smaller classes.
- Use progressive resizing while training.
- Use some field from metadata and build a multitask learning based classifier for both the product category as well as some metadata field.
- Combine image features and text features using a fusion network.

### Testing Ideas:
- Unit test corresponding to Trainer class that checks trains the model on a small number of epochs and compares the accuracy to the expected accuracy.
- Regression testing in Trainer class - to ensure that we don't regress performance
- Output tests: Obtain predictions for a fixed set of images and compare them against expected output.
