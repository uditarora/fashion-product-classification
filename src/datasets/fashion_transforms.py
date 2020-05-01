from torchvision import transforms

"""
Data augmentation and normalization for training and fine-tuning
Just normalization for test data
Original Image size = 1080x1440 for large dataset and 60x80 for small version
"""

def get_train_transforms(size=224):
    train_transforms = [
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(train_transforms)

def get_test_transforms(size=224):
    test_transforms = [
        transforms.Resize(int(size*1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(test_transforms)

def get_data_transforms(phase, size=224):
    if phase == 'train':
        return get_train_transforms(size=size)
    elif phase == 'val':
        return get_test_transforms(size=size)
    else:
        return get_test_transforms(size=size)
