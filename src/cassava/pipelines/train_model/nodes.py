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


def split_data(train_labels, parameters):
    """Splits trainig data into the train and validation set"""
    train_indices, val_indices = train_test_split(range(len(train_labels)),
                     stratify=train_labels.label,
                     random_state=parameters['seed'],
                     test_size=parameters['validation_size'])
    return train_indices, val_indices


def score_model(model, train_images_torch, indices, parameters):
    logging.debug('Scoring model')

    dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_torch, indices=indices),
                      transform=get_test_transforms())
    loader = torch.utils.data.DataLoader(dataset, num_workers=parameters['data_loader_workers'], batch_size=parameters['batch_size'])

    predictions = []
    true_labels = []
    model.eval()
    model.freeze()
    for images, labels in tqdm(loader):
        batch_preds = model.predict(images)
        predictions += batch_preds.tolist()
        true_labels += labels.tolist()

    scores = {
        'accuracy': accuracy_score(predictions, true_labels),
        'confusion_matrix': confusion_matrix(predictions, true_labels),
        'f1_score': f1_score(predictions, true_labels, average='weighted'),
    }

    logging.info(f'Validation scores:\n{scores}')
    return scores


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


def report_on_training(train_metrics):
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=range(len(train_metrics['train_losses'])), y=train_metrics['train_losses'], label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.lineplot(x=range(len(train_metrics['validation_losses'])), y=train_metrics['validation_losses'], label='Validation loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.lineplot(x=range(len(train_metrics['train_epoch_losses'])), y=train_metrics['train_epoch_losses'], label='Training loss per epoch')
    sns.lineplot(x=range(len(train_metrics['validation_epoch_losses'])), y=train_metrics['validation_epoch_losses'], label='Validation loss per epoch')
    plt.legend()
    plt.show()
