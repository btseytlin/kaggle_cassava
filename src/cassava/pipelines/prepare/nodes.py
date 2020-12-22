import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import logging
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from cassava.lmdb_dataset import dataset_to_lmdb
from cassava.transforms import lmdb_transforms
from cassava.utils import DatasetFromSubset
from PIL import Image
import imagehash


def get_img_hash(image):
    funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
    ]
    return np.array([f(image).hash for f in funcs]).reshape(192)


def obtain_image_hashes(train_images_torch_2020, train_images_torch_2019, test_images_torch_2019, extra_images_torch_2019):
    # Adapted from https://www.kaggle.com/zzy990106/duplicate-images-in-two-competitions
    datasets = {
        'train_2020': train_images_torch_2020,
        'train_2019': train_images_torch_2019,
        'test_2019': test_images_torch_2019,
        'extra_2019': extra_images_torch_2019,
    }

    image_ids = []
    hashes = []

    logging.info('Obtaining hashes')

    for dname, ds in tqdm(datasets.items()):
        loader = DataLoader(ds, num_workers=6, batch_size=None)
        for ix, (image, label) in tqdm(enumerate(loader), total=len(loader), desc=dname):
            if dname in ['test_2019', 'extra_2019']:
                label = None

            if label is not None:
                label = int(label)
            img_id = (dname, ix, label)
            pil_img = Image.fromarray(np.array(image))
            hash = get_img_hash(pil_img)
            image_ids.append(img_id)
            hashes.append(hash)

    image_ids_df = pd.DataFrame(image_ids, columns=['ds', 'ix', 'label'])
    hashes_df = pd.DataFrame(np.array(hashes).astype(int))

    return image_ids_df, hashes_df


def find_duplicates(image_ids, image_hashes):
    # Adapted from https://www.kaggle.com/zzy990106/duplicate-images-in-two-competitions
    image_ids = image_ids.values
    hashes = image_hashes.values

    hashes_all = np.array(hashes)
    hashes_all = torch.Tensor(hashes_all.astype(int))

    logging.info('Computing similarities and finding duplicates')
    sim_threshold = int(0.9 * hashes_all.shape[1])
    duplicates = []
    for i in tqdm(range(hashes_all.shape[0])):
        sim = ((hashes_all[i] == hashes_all).sum(dim=1).numpy() > sim_threshold).astype(int)
        dupes = np.nonzero(sim)[0]
        if len(dupes) > 1:
            for dup in dupes:
                if dup != i:
                    duplicates.append(tuple(sorted([i, dup])))

    duplicates = list(set(duplicates))

    out_rows = []
    for duplicate_pair in duplicates:
        image_id1 = image_ids[duplicate_pair[0]]
        image_id2 = image_ids[duplicate_pair[1]]
        out_rows.append(
            # ds1 | id1 | label1 | ds2 | id2 | label2
            (*image_id1, *image_id2)
        )

    out_rows = pd.DataFrame(list(set(out_rows)), columns=['ds1', 'id1', 'label1', 'ds2', 'id2', 'label2'])
    return out_rows


def prepare_lmdb(train_images_torch_2020, train_images_torch_2019, test_images_torch_2019, extra_images_torch_2019, duplicates):
    blacklist = dict(duplicates[['ds2', 'id2']].groupby('ds2').agg({'id2': list})['id2'])

    train_dataset_2020 = DatasetFromSubset(
        Subset(train_images_torch_2020, indices=[i for i in range(len(train_images_torch_2020)) if i not in blacklist['train_2020']]),
        transform=lmdb_transforms)

    train_dataset_2019 = DatasetFromSubset(
        Subset(train_images_torch_2019,
               indices=[i for i in range(len(train_images_torch_2019)) if i not in blacklist['train_2019']]),
        transform=lmdb_transforms)

    test_dataset_2019 = DatasetFromSubset(
        Subset(test_images_torch_2019,
               indices=[i for i in range(len(test_images_torch_2019)) if i not in blacklist['test_2019']]),
        transform=lmdb_transforms, target_transform=lambda y: -1)

    extra_images_torch_2019 = DatasetFromSubset(
        Subset(extra_images_torch_2019,
               indices=[i for i in range(len(extra_images_torch_2019)) if i not in blacklist['extra_2019']]),
        transform=lmdb_transforms, target_transform=lambda y: -1)

    train_dataset = ConcatDataset([train_dataset_2020, train_dataset_2019])
    train_sources = ['train_2020']*len(train_dataset_2020) + ['train_2019']*len(train_dataset_2019)

    unlabelled_dataset = ConcatDataset([test_dataset_2019, extra_images_torch_2019])
    unlabelled_sources = ['test_2019'] * len(test_dataset_2019) + ['extra_2019'] * len(extra_images_torch_2019)

    train_lmdb_path = '/data/cassava_extra_data/data/03_primary/train.lmdb'
    unlabelled_lmdb_path = '/data/cassava_extra_data/data/03_primary/unlabelled.lmdb'

    if any([os.path.exists(train_lmdb_path),
            os.path.exists(unlabelled_lmdb_path)]):
        raise Exception('LMDB files already exist, delete manually to overwrite.')

    train_lmdb = dataset_to_lmdb(train_dataset, train_sources, train_lmdb_path)
    unlabelled_lmdb = dataset_to_lmdb(unlabelled_dataset, unlabelled_sources, unlabelled_lmdb_path)

    return train_lmdb, unlabelled_lmdb
