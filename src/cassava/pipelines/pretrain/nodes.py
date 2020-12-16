import logging
from argparse import Namespace
import torch
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from cassava.models.model import LeafDoctorModel
from cassava.models.byol import BYOL
from cassava.node_helpers import lr_find, train_byol

from cassava.transforms import dummy_transforms


def pretrain_model(train_images_lmdb, test_images_lmdb, parameters):
    train_images_lmdb.transform = dummy_transforms
    test_images_lmdb.transform = dummy_transforms
    dataset = torch.utils.data.ConcatDataset([train_images_lmdb, test_images_lmdb])
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=parameters['byol']['batch_size'],
                                        num_workers=parameters['data_loader_workers'],
                                        shuffle=True)

    classifier_params = Namespace(**parameters['classifier'])
    model = LeafDoctorModel(classifier_params)

    hparams = Namespace(**parameters['byol'])
    byol = train_byol(model.trunk, hparams, loader)

    state_dict = byol.encoder.model.state_dict()
    model = LeafDoctorModel(classifier_params)
    model.trunk.load_state_dict(state_dict)
    return model

