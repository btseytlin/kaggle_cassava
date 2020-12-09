import logging
from argparse import Namespace

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from cassava.models.model import LeafDoctorModel
from cassava.transforms import get_train_transforms, get_test_transforms
from cassava.utils import DatasetFromSubset
from cassava.pipelines.predict.nodes import predict
from cassava.node_helpers import score


def split_data(train_labels, parameters):
    """Splits trainig data into the train and validation set"""
    train_indices, val_indices = train_test_split(range(len(train_labels)),
                     stratify=train_labels.label,
                     random_state=parameters['seed'],
                     test_size=parameters['validation_size'])
    return train_indices, val_indices


def score_model(model, train_images_torch, indices, parameters):
    logging.info('Scoring model')
    labels = train_images_torch.labels[indices]
    predictions = predict(model,
                          dataset=train_images_torch,
                          indices=indices,
                          batch_size=parameters['batch_size'],
                          num_workers=parameters['data_loader_workers'],
                          transform=get_test_transforms())

    scores = score(predictions, labels)

    logging.info(f'Validation scores:\n{scores}')
    return scores, predictions


def train_model(train_images_torch, train_indices, val_indices, parameters):
    train_transform, val_transform = get_train_transforms(), get_test_transforms()

    train_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_torch, indices=train_indices),
                                      transform=train_transform)

    val_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_torch, indices=val_indices),
                                    transform=val_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=parameters['batch_size'],
                                                    num_workers=parameters['data_loader_workers'],
                                                    shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, num_workers=parameters['data_loader_workers'], batch_size=parameters['batch_size'])

    # Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss",
                                       verbose=True,
                                       dirpath=parameters['checkpoints_dir'],
                                       filename="{epoch}_{val_loss:.4f}",
                                       save_top_k=parameters['save_top_k_checkpoints'])
    early_stopping = EarlyStopping('val_loss',
                                   patience=parameters['early_stop_patience'],
                                   verbose=True,
                                   )

    hparams = Namespace(**parameters)

    trainer = Trainer.from_argparse_args(
        hparams,
        reload_dataloaders_every_epoch = True,
        callbacks=[model_checkpoint, early_stopping],
    )

    # Model
    model = LeafDoctorModel(hparams)

    # LR finding
    # lr_finder = trainer.tuner.lr_find(model,
    #                                   train_dataloader=train_data_loader,
    #                                   val_dataloaders=[val_data_loader])
    # plt.figure()
    # plt.title('LR finder results')
    # lr_finder.plot(suggest=True)
    # plt.show()
    # new_lr = lr_finder.suggestion()
    #
    # logging.info('LR finder found this LR: %f', new_lr)
    # model.hparams.lr = new_lr

    # Training
    trainer.fit(model, train_data_loader, val_data_loader)
    logging.info('Training finished')

    # Saving
    best_checkpoint = model_checkpoint.best_model_path
    model = LeafDoctorModel().load_from_checkpoint(checkpoint_path=best_checkpoint)
    return model
