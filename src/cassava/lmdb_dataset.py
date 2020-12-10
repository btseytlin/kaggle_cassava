import logging
import os
from PIL import Image
import six

from torch.utils.data import DataLoader

import lmdb
from tqdm.auto import tqdm
import pyarrow as pa
import lz4framed

import torch.utils.data as data


def compress_serialize(thing):
    return pa.serialize(thing).to_buffer()


def deserialize_decompress(thing):
    return pa.deserialize(thing)


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


class ImageLMDBDataset(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = str(db_path)
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(db_path),
                                     readonly=True, lock=False,
                                     readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = deserialize_decompress(txn.get(b'__len__'))
            self.keys = deserialize_decompress(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = deserialize_decompress(byteflow)
        image, label = unpacked

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def dataset_to_lmdb(dataset, out_path, write_frequency=2000, num_workers=8, map_size=1e11):
    dataset.loader = raw_reader
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = out_path
    isdir = os.path.isdir(lmdb_path)

    logging.debug("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                           map_size=map_size, readonly=False,
                           meminit=False, map_async=True)

    labels = []
    logging.debug(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), compress_serialize((image, label)))
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)
        labels.append(label)

    # finish iterating through dataset
    logging.debug('Final commit')
    txn.commit()

    logging.debug('Writing keys and len')
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', compress_serialize(keys))
        txn.put(b'__len__', compress_serialize(len(keys)))

    logging.debug("Flushing database ...")
    db.sync()
    db.close()

    return ImageLMDBDataset(out_path)
