import logging
import numpy as np
from argparse import Namespace
import torch
from torch.utils.data import Subset, DataLoader

from cassava.models.model import LeafDoctorModel
from cassava.node_helpers import lr_find, train_byol, train_classifier

from cassava.transforms import get_byol_transforms, get_train_transforms
from cassava.utils import DatasetFromSubset


def finetune_byol_test(pretrained_model, train, test_images_torch_2020, parameters):
    byol_transforms = get_byol_transforms(parameters['byol']['width'], parameters['byol']['height'])

    train_2020_indices = np.argwhere(train.sources == 'train_2020').flatten()
    train_2020 = DatasetFromSubset(Subset(train, indices=train_2020_indices))
    dataset = torch.utils.data.ConcatDataset([train_2020, test_images_torch_2020])
    dataset = DatasetFromSubset(
        torch.utils.data.Subset(dataset, indices=list(range(len(dataset)))),
        transform = byol_transforms)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=parameters['byol']['batch_size'],
                                         num_workers=parameters['data_loader_workers'],
                                         shuffle=True,
                                         pin_memory=True)

    byol_params = dict(parameters['byol'])
    byol_test_overrides = dict(parameters['byol']['on_test'])
    byol_params.update(byol_test_overrides)

    finetuned_model = train_byol(pretrained_model, loader,
                                  byol_parameters=byol_params,
                                  log_training=parameters['log_training'],
                                  logger_name='byol_test')
    return finetuned_model


def finetune_classifier_resolution(model, train, parameters):
    logging.info('Finetuning model for test image size')

    train_2020_indices = np.argwhere(train.sources == 'train_2020').flatten()
    train_2020 = DatasetFromSubset(Subset(train, indices=train_2020_indices))

    train_transform = get_train_transforms(parameters['classifier']['test_width'],
                                           parameters['classifier']['test_height'])
    train_dataset = DatasetFromSubset(Subset(train_2020, indices=list(range(len(train_2020)))),
                                      transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=parameters['classifier']['batch_size'],
                              num_workers=parameters['data_loader_workers'],
                              shuffle=True,
                              pin_memory=True)

    hparams = dict(parameters['classifier'])
    hparams.update(dict(parameters['classifier']['finetune']))
    hparams = Namespace(**hparams)

    only_train_layers = [
        lambda trunk: trunk.blocks[-1],
        lambda trunk: trunk.conv_head,
        lambda trunk: trunk.bn2,
        lambda trunk: trunk.global_pool,
        lambda trunk: trunk.act2,
        lambda trunk: trunk.classifier,
    ]
    model = train_classifier(model, train_loader,
                             hparams=hparams,
                             only_train_layers=only_train_layers,
                             log_training=parameters['log_training'],
                             logger_name='classifier_finetune')
    return model
