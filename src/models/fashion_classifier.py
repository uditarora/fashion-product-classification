import torch 
import torch.nn as nn 
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class FashionClassifier(nn.Module):
    """
    A model that will be used to classify the fashion images.

    It uses a pretrained Resnet50 model as the base, and changes the final
    linear layer to output probabilities corresponding to the number of classes

    Args:
        - num_classes (int): The number of classes in the output layer
        - model: The base model with resnet + fc output layer
        - weight (numpy array): Class weights to be used in loss function
    """
    def __init__(self, num_classes=20, model=None, weight=None):
        super(FashionClassifier, self).__init__()
        if model is not None:
            self.resnet = model.resnet
            num_features = model.fc.in_features
        else:
            self.resnet = models.resnet50(pretrained=True)
            num_features = self.resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)
        self.resnet.fc = Identity()
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight))

    def forward(self, x, y=None):
        """
        Takes the inputs (x) and labels (y) as input and returns the output
        and loss (if y is provided)
        """
        x = self.resnet(x)
        x = self.fc(x)

        if y is not None:
            loss = self.criterion(x, y)
            return x, loss
        else:
            return x

class FashionClassifierMT(nn.Module):
    """
    A model that will be used to classify the fashion images.

    It uses a pretrained Resnet50 model as the base, and changes the final
    linear layer to output probabilities corresponding to number of classes.
    An auxiliary task is added to improve the performance.

    Args:
        - num_classes (int): The number of classes in the output layer
        - model: The base model with resnet + fc output layer
        - weight (numpy array): Class weights to be used in loss function
    """
    def __init__(self, num_classes=20, model=None, weight=None):
        super(FashionClassifierMT, self).__init__()
        if model is not None:
            self.resnet = model.resnet
            num_features = model.fc.in_features
        else:
            self.resnet = models.resnet50(pretrained=True)
            num_features = self.resnet.fc.in_features
        self.resnet.fc = Identity()
        self.fc = nn.Linear(num_features, num_classes)
        self.fc2 = nn.Linear(num_features, 7)
        self.fc3 = nn.Linear(num_features, 45)

        self.criterion1 = nn.CrossEntropyLoss(weight=torch.Tensor(weight))
        self.criterion2 = nn.CrossEntropyLoss()

    def forward(self, x, ys=None):
        """
        Takes the inputs (x) and labels (ys) as input and returns the outputs
        and loss (if ys is provided)
        """
        x = self.resnet(x)
        x1 = self.fc(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)

        if ys is not None:
            loss = 2*self.criterion1(x1, ys[0])
            loss += self.criterion2(x2, ys[1])/2
            loss += self.criterion2(x3, ys[2])/2
            return (x1, x2, x3), loss
        else:
            return (x1, x2, x3)


def get_top20_classifier(weight=None, mt=False):
    if mt:
        return FashionClassifierMT(num_classes=20, weight=weight)
    else:
        return FashionClassifier(num_classes=20, weight=weight)

def get_ft_classifier(top20_classifier=None, weight=None, mt=False):
    if mt:
        return FashionClassifierMT(num_classes=122, model=top20_classifier,
                                   weight=weight)
    else:
        return FashionClassifier(num_classes=122, model=top20_classifier,
                                 weight=weight)
