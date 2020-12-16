import logging
from argparse import Namespace
import torch

from cassava.models.model import LeafDoctorModel
from cassava.node_helpers import lr_find, train_byol

from cassava.transforms import dummy_transforms


def finetune_on_test(pretrained_model, train_images_lmdb, test_images_lmdb, parameters):
    train_images_lmdb.transform = dummy_transforms
    test_images_lmdb.transform = dummy_transforms
    dataset = torch.utils.data.ConcatDataset([train_images_lmdb, test_images_lmdb])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=parameters['byol']['batch_size'],
                                         num_workers=parameters['data_loader_workers'],
                                         shuffle=True)
    classifier_params = Namespace(**parameters['classifier'])
    byol_params = dict(parameters['byol'])
    byol_test_overrides = dict(parameters['byol']['on_test'])
    byol_params.update(byol_test_overrides)

    hparams = Namespace(**byol_params)

    byol = train_byol(pretrained_model.trunk, hparams, loader)

    state_dict = byol.encoder.model.state_dict()
    model = LeafDoctorModel(classifier_params)
    model.trunk.load_state_dict(state_dict)
    return model
