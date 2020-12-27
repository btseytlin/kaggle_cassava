import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

byol_transforms = A.Compose([
    A.Resize(360, 360),
    A.ToFloat(max_value=1.0),
    ToTensorV2(),
])

prepare_transforms = A.Compose([
])


def get_train_transforms():
    return A.Compose([
        A.Resize(400, 400),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
        A.JpegCompression(quality_lower=95, quality_upper=100, p=0.5),
        A.ColorJitter(p=0.5),
        A.ToFloat(max_value=1.0),
        A.ShiftScaleRotate(p=0.5),
        A.RandomResizedCrop(360, 360, scale=(0.1, 0.8)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.CoarseDropout(p=0.5),
        ToTensorV2(),
    ])


def get_test_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.ToFloat(max_value=1.0),
        A.CenterCrop(450, 450),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

