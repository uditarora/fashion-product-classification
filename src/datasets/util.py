import numpy as np

def get_class_weights(df, classmap, eps=1):
    """
    Returns class weights corresponding to each class using the formula:
        weights = n_samples / (n_classes * (np.bincount(y)+eps))
    'eps' is added to handle classes that don't have any samples
    """
    labels = [classmap[x] for x in df['articleType']]
    labels_count = np.bincount(labels) + eps
    return len(labels) / (len(classmap) * labels_count)
