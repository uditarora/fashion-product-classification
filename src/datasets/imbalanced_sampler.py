"""
Adapted from https://github.com/ufoym/imbalanced-dataset-sampler
"""
import torch
import torch.utils.data
import torchvision

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        dataset (Dataset): the dataset object
        weights (np.array): class weights in the dataset
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """
    def __init__(self, dataset, weights, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
