import logging
from argparse import Namespace

from torch.utils.data import ConcatDataset, Subset, DataLoader

from cassava.models.model import LeafDoctorModel
from cassava.node_helpers import train_byol

from cassava.transforms import byol_transforms
from cassava.utils import DatasetFromSubset


def pretrain_model(train, unlabelled, parameters):
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
                        shuffle=True)

    classifier_params = Namespace(**parameters['classifier'])
    model = LeafDoctorModel(classifier_params)

    hparams = Namespace(**parameters['byol'])
    byol = train_byol(model.trunk, hparams, loader)

    state_dict = byol.encoder.model.state_dict()
    model = LeafDoctorModel(classifier_params)
    model.trunk.load_state_dict(state_dict)
    return model

