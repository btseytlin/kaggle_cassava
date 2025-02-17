import logging
from argparse import Namespace
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

from cassava.transforms import get_train_transforms, get_test_transforms
from cassava.utils import DatasetFromSubset
from cassava.pipelines.predict.nodes import predict
from cassava.node_helpers import score, train_classifier


def split_data(train, parameters):
    """Splits trainig data into the train and validation set"""
    labels = train.labels
    train_indices, val_indices = train_test_split(range(len(labels)),
                     stratify=labels,
                     random_state=parameters['seed'],
                     test_size=parameters['validation_size'])
    return train_indices, val_indices


def train_model(pretrained_model, train, parameters):
    train_transform = get_train_transforms(parameters['classifier']['train_width'], parameters['classifier']['train_height'])

    train_dataset = DatasetFromSubset(Subset(train, indices=list(range(len(train)))),
                                      transform=train_transform)

    train_loader = DataLoader(train_dataset,
                                batch_size=parameters['classifier']['batch_size'],
                                num_workers=parameters['data_loader_workers'],
                                shuffle=True,
                              pin_memory=True)

    hparams = Namespace(**parameters['classifier'])

    # Train
    logging.info('Training model')
    model = train_classifier(pretrained_model, train_loader, hparams=hparams)
    return model


def score_model(model, train, indices, parameters):
    logging.info('Scoring model')
    if parameters['classifier'].get('limit_val_batches'):
        indices = indices[:parameters['classifier']['limit_val_batches']*parameters['classifier']['batch_size']]
    labels = np.array(train.labels)[indices]
    predictions, probas = predict(model,
                          dataset=train,
                          indices=indices,
                          batch_size=parameters['eval']['batch_size'],
                          num_workers=parameters['data_loader_workers'],
                          transform=get_test_transforms(parameters['classifier']['test_width'], parameters['classifier']['test_height']))

    scores = score(predictions, labels)

    logging.info(f'Validation scores:\n{scores}')
    return scores, predictions
