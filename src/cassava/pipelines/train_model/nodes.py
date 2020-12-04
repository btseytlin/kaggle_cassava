import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from cassava.models.resnet50 import ResnetModel

from cassava.transforms import get_train_transforms, get_test_transforms

from cassava.extras.datasets.image_dataset import DatasetFromSubset


def split_data(train_labels, parameters):
    """Splits trainig data into the train and validation set"""
    train_indices, val_indices = train_test_split(range(len(train_labels)),
                     stratify=train_labels.label,
                     random_state=parameters['seed'],
                     test_size=parameters['validation_size'])
    return train_indices, val_indices


def score_model(model, train_images_torch, indices, parameters):
    logging.debug('Scoring model')

    device = parameters['device']

    dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_torch, indices=indices),
                      transform=get_test_transforms())
    loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=parameters['batch_size'])

    predictions = []
    true_labels = []
    model.eval()
    model = model.to(device)
    for images, labels in tqdm(loader):
        batch_preds = model.predict_label(images.to(device))
        predictions += batch_preds.tolist()
        true_labels += labels.tolist()

    return {
        'accuracy': accuracy_score(predictions, true_labels),
        'confusion_matrix': confusion_matrix(predictions, true_labels),
        'f1_score': f1_score(predictions, true_labels, average='weighted'),
    }


def train_model(train_images_torch, train_indices, val_indices, parameters):
    train_transform, val_transform = get_train_transforms(), get_test_transforms()

    train_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_torch, indices=train_indices),
                                      transform=train_transform)

    val_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_torch, indices=val_indices),
                                    transform=val_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=parameters['batch_size'],
                                                    num_workers=8,
                                                    shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=parameters['batch_size'])

    model = ResnetModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=parameters['reduce_lr_on_pleteau_patience'], verbose=True)

    model = model.to(parameters['device'])
    criterion = criterion.to(parameters['device'])

    early_stop_patience = parameters['early_stop_patience']
    early_stop_counter = 0
    previous_min_val_loss = None

    train_losses = []
    validation_losses = []

    train_epoch_losses = []
    validation_epoch_losses = []

    logging.info('Training model')
    epoch_pbar = tqdm(range(parameters['train_epochs']))
    for epoch in epoch_pbar:
        model.train()

        logging.debug("Epoch %d", epoch)
        epoch_train_losses = []
        epoch_val_losses = []

        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for i, batch in pbar:
            if i > parameters['batches_per_epoch']:
                break
            inputs, labels = batch
            inputs = inputs.to(parameters['device'])
            labels = labels.to(parameters['device'])

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            pbar.set_postfix({'batch loss': round(loss.item(), 4)})

        model.eval()
        for i, batch in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
            if i > parameters['batches_per_epoch']:
                break
            with torch.no_grad():
                inputs, labels = batch
                inputs = inputs.to(parameters['device'])
                labels = labels.to(parameters['device'])

                outputs = model.forward(inputs)
                val_loss = criterion(outputs, labels)

                epoch_val_losses.append(val_loss.item())

        epoch_mean_val_loss = sum(epoch_val_losses)/len(epoch_val_losses)
        epoch_mean_train_loss = sum(epoch_train_losses)/len(epoch_train_losses)
        if previous_min_val_loss is None:
            previous_min_val_loss = epoch_mean_val_loss
        elif epoch_mean_val_loss < previous_min_val_loss:
            previous_min_val_loss = epoch_mean_val_loss
            early_stop_counter = 0
            logging.debug('New minimum val loss %f, early stopping reset', previous_min_val_loss)
        else:
            early_stop_counter += 1
            logging.debug('Early stop counter now %d', early_stop_counter)

        lr_scheduler.step(sum(epoch_train_losses))

        train_epoch_losses.append(epoch_mean_train_loss)
        validation_epoch_losses.append(epoch_mean_val_loss)
        logging.info("Epoch mean train loss %f", epoch_mean_train_loss)
        logging.info("Epoch mean val loss %f", epoch_mean_val_loss)

        epoch_pbar.set_postfix({
            'train loss': epoch_mean_train_loss,
            'val loss': epoch_mean_val_loss,
        })

        train_losses += epoch_train_losses
        validation_losses += epoch_val_losses

        if early_stop_counter >= early_stop_patience:
            logging.info('Early stopped.')
            break

    logging.info('Training finished')

    metrics = {
        'train_losses': train_losses,
        'validation_losses': validation_losses,
        'train_epoch_losses': train_epoch_losses,
        'validation_epoch_losses': validation_epoch_losses,
        'last_epoch': epoch,
    }

    return model, metrics

