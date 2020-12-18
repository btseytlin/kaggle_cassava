import logging
from argparse import Namespace
import torch

from cassava.models.model import LeafDoctorModel
from cassava.node_helpers import lr_find, train_byol

from cassava.transforms import dummy_transforms
from cassava.utils import DatasetFromSubset


def finetune_on_test(pretrained_model, train_images_lmdb, test_images_lmdb, parameters):
    dataset = torch.utils.data.ConcatDataset([train_images_lmdb, test_images_lmdb])
    dataset = DatasetFromSubset(
        torch.utils.data.Subset(dataset, indices=list(range(len(dataset)))),
        transform = dummy_transforms)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=parameters['byol']['batch_size'],
                                         num_workers=parameters['data_loader_workers'],
                                         shuffle=True)

    byol_params = dict(parameters['byol'])
    byol_test_overrides = dict(parameters['byol']['on_test'])
    byol_params.update(byol_test_overrides)

    hparams = Namespace(**byol_params)

    state_dict = pretrained_model.state_dict()
    model = LeafDoctorModel(Namespace(**parameters['classifier']))
    model.load_state_dict(state_dict)

    byol = train_byol(pretrained_model.trunk, hparams, loader)

    state_dict = byol.encoder.model.state_dict()
    model = LeafDoctorModel(Namespace(**parameters['classifier']))
    model.trunk.load_state_dict(state_dict)
    return model
