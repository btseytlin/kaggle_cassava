
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def get_train_transforms():
    return A.Compose([
        A.ToFloat(max_value=1.0),
        A.Resize(256, 256),
        A.RandomResizedCrop(227, 227, scale=(0.4, 1), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_transforms():
    return A.Compose([
        A.ToFloat(max_value=1.0),
        A.Resize(256, 256),
        A.CenterCrop(227, 227),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

