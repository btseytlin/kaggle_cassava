import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from cassava.models.resnet50 import ResnetModel


class Model(nn.Module):
    def __init__(self, trunk, head):
        super(Model, self).__init__()
        self.trunk = trunk
        self.trunk.fc = head
        self.head = self.trunk.fc

    def forward(self, x):
        return self.trunk.forward(x)

    def predict(self, x):
        logits = self.forward(x)
        probabilities = nn.functional.softmax(logits)
        return probabilities

    def predict_label(self, x):
        return torch.max(self.predict(x), 1)[1]


def split_data(train_labels, parameters):
    """Splits trainig data into the train and validation set"""
    train_indices, val_indices = train_test_split(range(len(train_labels)),
                     stratify=train_labels.label,
                     random_state=parameters['seed'],
                     test_size=parameters['validation_size'])
    return train_indices, val_indices


def train_model(train_images_torch, train_indices, val_indices, parameters):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(227, scale=(0.16, 1), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torch.utils.data.Subset(train_images_torch, indices=train_indices)
    train_dataset.transform = train_transform

    val_dataset = torch.utils.data.Subset(train_images_torch, indices=val_indices)
    val_dataset.transform = val_transform

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=parameters['batch_size'],
                                                    num_workers=4,
                                                    shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=parameters['batch_size'])

    model = ResnetModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

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

        for i, batch in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
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
