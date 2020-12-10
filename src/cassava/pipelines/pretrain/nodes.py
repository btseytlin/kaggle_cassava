import logging
from argparse import Namespace
import torch

from pytorch_lightning import Trainer

from cassava.models.model import LeafDoctorModel
from cassava.models.byol import BYOL
from cassava.transforms import get_test_transforms


def pretrain_model(train_images_lmdb, test_images_lmdb, parameters):
    transform = get_test_transforms()
    train_images_lmdb.transform = transform
    test_images_lmdb.transform = transform
    dataset = torch.utils.data.ConcatDataset([train_images_lmdb, test_images_lmdb])
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=parameters['byol']['batch_size'],
                                        num_workers=parameters['data_loader_workers'],
                                        shuffle=True)
    classifier_params = Namespace(**parameters['classifier'])
    model = LeafDoctorModel(classifier_params)
    byol = BYOL(model.trunk, image_size=(256, 256), **parameters['byol'])
    trainer = Trainer.from_argparse_args(
        Namespace(**parameters['byol']),
        reload_dataloaders_every_epoch=True,
        terminate_on_nan=True,
    )

    trainer.fit(byol, loader, loader)

    state_dict = byol.encoder.model.state_dict()
    model = LeafDoctorModel(classifier_params)
    model.trunk.load_state_dict(state_dict)
    return model
