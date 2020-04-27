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
    A model that will be used to classify the top-20 classes.

    It uses a pretrained Resnet50 model as the base, and changes the final
    linear layer to output probabilities corresponding to the top-20 classes
    """
    def __init__(self, num_classes=20, resnet=None):
        super(FashionClassifier,self).__init__()
        if resnet is not None:
            self.resnet = resnet
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
    return FashionClassifier(num_classes=122, resnet=top20_classifier.resnet)
