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

    classifier_params = Namespace(**parameters['classifier'])

    only_train_layers = [
        lambda trunk: trunk.blocks[-1],
        lambda trunk: trunk.conv_head,
        lambda trunk: trunk.bn2,
        lambda trunk: trunk.global_pool,
        lambda trunk: trunk.classifier,
    ]

    model = LeafDoctorModel(classifier_params, only_train_layers=only_train_layers)

    hparams = Namespace(**parameters['byol'])
    byol = train_byol(model.trunk, loader,
                      hparams=hparams,
                      log_training=parameters['log_training'])

    state_dict = byol.encoder.model.state_dict()
    model = LeafDoctorModel(classifier_params)
    model.trunk.load_state_dict(state_dict)
    return model

