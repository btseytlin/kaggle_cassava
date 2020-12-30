import logging
from argparse import Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Subset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from cassava.models.byol import BYOL
from cassava.models.model import LeafDoctorModel
from cassava.transforms import get_test_transforms
from cassava.utils import DatasetFromSubset
from matplotlib import pyplot as plt


def score(predictions, labels):
    return {
        'accuracy': accuracy_score(predictions, labels),
        'f1_score': f1_score(predictions, labels, average='weighted'),
    }


def predict(model, dataset, indices, batch_size=10, num_workers=4, transform=None):
    dataset = DatasetFromSubset(
        Subset(dataset, indices=indices),
        transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True)

    predictions = []
    probas = []
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            if torch.cuda.is_available():
                images = images.cuda()
            batch_probas = model.predict_proba(images)
            batch_preds = torch.max(batch_probas, 1)[1]
            predictions.append(batch_preds)
            probas.append(batch_probas)

    predictions = torch.hstack(predictions).flatten().tolist()
    probas = torch.vstack(probas).tolist()

    return predictions, probas


def lr_find(trainer, model, train_data_loader, val_data_loader=None, plot=False):
    val_dataloaders = [val_data_loader] if val_data_loader else None

    lr_finder = trainer.tuner.lr_find(model,
                                      train_dataloader=train_data_loader,
                                      val_dataloaders=val_dataloaders)
    if plot:
        plt.figure()
        plt.title('LR finder results')
        lr_finder.plot(suggest=True)
        plt.show()

    newlr = lr_finder.suggestion()
    logging.info('LR finder suggestion: %f', newlr)

    return newlr


def train_classifier(model, train_loader, hparams, only_train_layers=None, log_training=True, logger_name='classifier'):
    logger = TensorBoardLogger("lightning_logs", name=logger_name) if log_training else None
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(
        hparams,
        reload_dataloaders_every_epoch=True,
        terminate_on_nan=True,
        precision=hparams.precision,
        amp_level=hparams.amp_level,
        callbacks=[lr_monitor] if log_training else None,
        log_every_n_steps=hparams.log_every_n_steps,
        flush_logs_every_n_steps=hparams.flush_logs_every_n_steps,
        logger=logger,
    )

    # Model
    new_model = LeafDoctorModel(hparams, only_train_layers=only_train_layers)
    new_model.load_state_dict(model.state_dict())
    model = new_model

    # Training
    trainer.fit(model, train_loader)
    logging.info('Training finished')
    return model


def train_byol(model, loader, byol_parameters, log_training=True, logger_name='byol'):
    only_train_layers = [
        lambda trunk: trunk.blocks[-1],
        lambda trunk: trunk.conv_head,
        lambda trunk: trunk.bn2,
        lambda trunk: trunk.global_pool,
        lambda trunk: trunk.act2,
        lambda trunk: trunk.classifier,
    ]
    new_model = LeafDoctorModel(only_train_layers=only_train_layers)
    new_model.load_state_dict(model.state_dict())
    model = new_model

    hparams = Namespace(**byol_parameters)

    logger = TensorBoardLogger("lightning_logs", name=logger_name) if log_training else None
    byol = BYOL(model.trunk, hparams=hparams)
    early_stopping = EarlyStopping('train_loss',
                                   mode='min',
                                   patience=hparams.early_stop_patience,
                                   verbose=True)
    callbacks = [early_stopping]
    if log_training:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    trainer = Trainer.from_argparse_args(
        hparams,
        reload_dataloaders_every_epoch=True,
        terminate_on_nan=True,
        callbacks=callbacks,
        precision=hparams.precision,
        amp_level=hparams.amp_level,
        log_every_n_steps=hparams.log_every_n_steps,
        flush_logs_every_n_steps=hparams.flush_logs_every_n_steps,
        logger=logger,
    )

    if hparams.auto_lr_find:
        new_lr = lr_find(trainer, byol, loader)
        hparams.lr = new_lr
        byol.hparams.lr = new_lr

    trainer.fit(byol, loader, loader)

    pretrained_model = LeafDoctorModel(None)
    pretrained_model.trunk.load_state_dict(byol.encoder.model.state_dict())
    return pretrained_model
