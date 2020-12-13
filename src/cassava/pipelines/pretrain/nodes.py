import logging
from argparse import Namespace
import torch
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from cassava.models.model import LeafDoctorModel
from cassava.models.byol import BYOL
from cassava.node_helpers import lr_find
import albumentations as A
from cassava.transforms import get_test_transforms


def pretrain_model(train_images_lmdb, test_images_lmdb, parameters):
    transforms = A.Compose([
        A.ToFloat(max_value=1.0),
        ToTensorV2(),
    ])

    train_images_lmdb.transform = transforms
    test_images_lmdb.transform = transforms
    dataset = torch.utils.data.ConcatDataset([train_images_lmdb, test_images_lmdb])
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=parameters['byol']['batch_size'],
                                        num_workers=parameters['data_loader_workers'],
                                        shuffle=True)

    classifier_params = Namespace(**parameters['classifier'])
    model = LeafDoctorModel(classifier_params)

    hparams = Namespace(**parameters['byol'])

    if hparams.from_checkpoint:
        logging.warning("Pretraining from checkpoint")
        model.load_state_dict(torch.load('data/06_models/pretrained_model.pt'))

    byol = BYOL(model.trunk, image_size=(256, 256), hparams=hparams)
    byol.cuda()

    early_stopping = EarlyStopping('train_loss',
                                   patience=parameters['byol']['early_stop_patience'],
                                   verbose=True)

    trainer = Trainer.from_argparse_args(
        hparams,
        reload_dataloaders_every_epoch=True,
        terminate_on_nan=True,
        callbacks=[early_stopping],
    )

    if hparams.auto_lr_find:
        new_lr = lr_find(trainer, model, loader)
        hparams.lr = new_lr
        model.hparams.lr = new_lr

    trainer.fit(byol, loader, loader)

    state_dict = byol.encoder.model.state_dict()
    model = LeafDoctorModel(classifier_params)
    model.trunk.load_state_dict(state_dict)
    return model
