import os
import torch
from cassava.lmdb_dataset import dataset_to_lmdb
from cassava.transforms import lmdb_transforms
from cassava.utils import DatasetFromSubset


def prepare_lmdb(train_images_torch, test_images_torch):
    train_dataset = DatasetFromSubset(
        torch.utils.data.Subset(train_images_torch, indices=list(range(len(train_images_torch)))),
        transform=lmdb_transforms)

    test_dataset = DatasetFromSubset(
        torch.utils.data.Subset(test_images_torch, indices=list(range(len(test_images_torch)))),
        transform=lmdb_transforms)

    train_lmdb_path = 'data/03_primary/train.lmdb'
    test_lmdb_path = 'data/03_primary/test.lmdb'

    if any([os.path.exists(train_lmdb_path),
            os.path.exists(test_lmdb_path)]):
        raise Exception('LMDB files lready exist, delete manually to overwrite.')

    train_images_lmdb = dataset_to_lmdb(train_dataset, train_lmdb_path)
    test_images_lmdb = dataset_to_lmdb(test_dataset, test_lmdb_path)

    return train_images_lmdb, test_images_lmdb