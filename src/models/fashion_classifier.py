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
    """
    def __init__(self, num_classes=20, model=None):
        super(FashionClassifier,self).__init__()
        if model is not None:
            self.resnet = model.resnet
            num_features = model.fc.in_features
        else:
            self.resnet = models.resnet50(pretrained=True)
            num_features = self.resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)
        self.resnet.fc = Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

def get_top20_classifier():
    return FashionClassifier(num_classes=20)

def get_ft_classifier(top20_classifier=None):
    return FashionClassifier(num_classes=122, model=top20_classifier)
