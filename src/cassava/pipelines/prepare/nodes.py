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

    train_images_lmdb = dataset_to_lmdb(train_images_torch, train_lmdb_path)
    test_images_lmdb = dataset_to_lmdb(test_images_torch, test_lmdb_path)

