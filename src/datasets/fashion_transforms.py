import Augmentor
from torchvision import transforms

# Data augmentation and normalization for training and fine-tuning
# Just normalization for test data

train_transforms = [
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

test_transforms = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

val_transforms = test_transforms

p = Augmentor.Pipeline()
p.rotate(probability=0.6, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.5)
p.zoom(probability=0.4, min_factor=1, max_factor=1.2)
p.shear(probability=0.8, max_shear_left=10, max_shear_right=10)
p.skew(probability=0.8)

train_small_transforms = [
    p.torch_transform(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

def get_data_transforms(phase, small=False):
    if phase == 'train':
        if small:
            return transforms.Compose(train_small_transforms)
        else:
            return transforms.Compose(train_transforms)
    elif phase == 'val':
        return transforms.Compose(val_transforms)
    else:
        return transforms.Compose(test_transforms)
