import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

byol_transforms = A.Compose([
    A.Resize(256, 256),
    A.ToFloat(max_value=1.0),
    ToTensorV2(),
])

lmdb_transforms = A.Compose([
    A.Resize(512, 512),
])


def get_train_transforms():
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
        A.ToFloat(max_value=1.0),
        A.RandomResizedCrop(256, 256, scale=(0.3, 0.9)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_transforms():
    return A.Compose([
        A.ToFloat(max_value=1.0),
        A.Resize(400, 400),
        A.CenterCrop(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

