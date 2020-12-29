import logging
from argparse import Namespace

from torch.utils.data import ConcatDataset, Subset, DataLoader

from cassava.models.model import LeafDoctorModel
from cassava.node_helpers import train_byol

from cassava.transforms import get_byol_transforms
from cassava.utils import DatasetFromSubset


def pretrain_model(train, unlabelled, parameters):
    byol_transforms = get_byol_transforms(parameters['byol']['width'], parameters['byol']['height'])
    train_dataset = DatasetFromSubset(
        Subset(train, indices=list(range(len(train)))),
        transform=byol_transforms)
    unlabelled_dataset = DatasetFromSubset(
        Subset(unlabelled, indices=list(range(len(unlabelled)))),
        transform=byol_transforms)
    dataset = ConcatDataset([train_dataset, unlabelled_dataset])
    loader = DataLoader(dataset,
                        batch_size=parameters['byol']['batch_size'],
                        num_workers=parameters['data_loader_workers'],
                        shuffle=True,
                        pin_memory=True)

    byol_params = parameters['byol']
    model = LeafDoctorModel()
    pretrained_model = train_byol(model, loader,
                                  byol_parameters=byol_params,
                                  log_training=parameters['log_training'],
                                  logger_name='byol_train')
    return pretrained_model

