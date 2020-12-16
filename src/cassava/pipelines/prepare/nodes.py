import os
from cassava.lmdb_dataset import dataset_to_lmdb
from cassava.transforms import lmdb_transforms


def prepare_lmdb(train_images_torch, test_images_torch):
    train_images_torch.transform = lmdb_transforms
    test_images_torch.transform = lmdb_transforms

    train_lmdb_path = 'data/03_primary/train.lmdb'
    test_lmdb_path = 'data/03_primary/test.lmdb'

    if any([os.path.exists(train_lmdb_path),
            os.path.exists(test_lmdb_path)]):
        raise Exception('LMDB files lready exist, delete manually to overwrite.')

    train_images_lmdb = dataset_to_lmdb(train_images_torch, train_lmdb_path)
    test_images_lmdb = dataset_to_lmdb(test_images_torch, test_lmdb_path)

    return train_images_lmdb, test_images_lmdb
