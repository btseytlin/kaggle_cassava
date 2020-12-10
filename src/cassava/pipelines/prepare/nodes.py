import os
from cassava.lmdb_dataset import dataset_to_lmdb
import albumentations as A


raw_transforms = A.Compose([
    A.Resize(400, 400),
])


def prepare_lmdb(train_images_torch, test_images_torch):
    train_images_torch.transform = raw_transforms
    test_images_torch.transform = raw_transforms

    train_lmdb_path = 'data/03_primary/train.lmdb'
    test_lmdb_path = 'data/03_primary/test.lmdb'

    if any([os.path.exists(train_lmdb_path),
            os.path.exists(test_lmdb_path)]):
        raise Exception('LMDB files lready exist, delete manually to overwrite.')

    dataset_to_lmdb(train_images_torch, train_lmdb_path)
    dataset_to_lmdb(test_images_torch, test_lmdb_path)

