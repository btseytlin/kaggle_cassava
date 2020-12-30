import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def data_preapre_transform(image):
    """Transforms a PIL Image to have aspec ratio 8/6"""
    image = Image.fromarray(np.array(image))
    if image.size[0] < image.size[1]:
        image = image.rotate(90, expand=True)

    # Center crop until 8:6
    width, height = image.size  # Get dimensions
    if round(width / height, 3) != round(8 / 6, 3):
        new_height = int(height * (width / height * 6 / 8))
        new_width = width

        if new_height > height:
            new_height = height
            new_width = int(width * 8 / 6 * height / width)

        left = abs(new_width - width) / 2
        top = abs(new_height - height) / 2
        right = (width + new_width) / 2
        bottom = (new_height + height) / 2

        assert (np.array([left, top, right, bottom]) >= 0).all()
        image = image.crop((left, top, right, bottom))
    return image


def get_wrapper(transforms):
    def wraps(img):
        return transforms(image=np.array(img))['image']
    return wraps


def get_byol_transforms(width, height):
    byol_transforms = A.Compose([
        A.Resize(width, height),
        A.ToFloat(max_value=1.0),
        ToTensorV2(),
    ])

    return get_wrapper(byol_transforms)


def get_prepare_transforms(width, height):
    prepare_transforms = A.Compose([
        A.Resize(width, height),
    ])

    return get_wrapper(prepare_transforms)


def get_train_transforms(width, height):
    train_transforms = A.Compose([
        A.RandomResizedCrop(width, height, scale=(0.1, 0.8)),
        A.JpegCompression(quality_lower=95, quality_upper=100, p=0.5),
        A.ColorJitter(p=0.5),
        A.ToFloat(max_value=1.0),
        A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.CoarseDropout(p=0.5),
        A.Cutout(p=0.5),
        ToTensorV2(),
    ])

    return get_wrapper(train_transforms)


def get_test_transforms(width, height):
    test_transforms = A.Compose([
        A.ToFloat(max_value=1.0),
        A.CenterCrop(width, height),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return get_wrapper(test_transforms)

