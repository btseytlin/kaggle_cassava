import os
import logging
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import Subset

from cassava.pipelines.finetune.nodes import finetune_byol_test, finetune_classifier_resolution
from cassava.transforms import get_test_transforms, data_preapre_transform, get_prepare_transforms
from cassava.models.model import LeafDoctorModel
from cassava.node_helpers import predict
from cassava.utils import DatasetFromSubset, make_image_folder, CassavaDataset


def prepare_test_dataset(test_images_torch_2020):
    test_images_torch_2020.transform = data_preapre_transform
    prepare_transforms = get_prepare_transforms(512, 512)

    test_dataset_2020 = DatasetFromSubset(
        Subset(test_images_torch_2020, indices=list(range(len(test_images_torch_2020)))),
        transform=prepare_transforms)

    test_sources = ['test_2020']*len(test_dataset_2020)

    path = 'data/03_primary/test'
    csv_path = 'data/03_primary/test.csv'

    if os.path.exists(path):
        raise Exception('Test dataset folder already exists, delete manually to overwrite.')

    test_df = make_image_folder(test_dataset_2020, test_sources, path, csv_path)
    return CassavaDataset(path, test_df.image_id, test_df.label, sources=test_sources)


def predict_submission(cv_results, train, test, sample_submission, parameters):
    def finetune_cv_model(model, train, parameters):
        logging.info('Finetuning for test resolution')
        model = finetune_classifier_resolution(model, train, parameters)
        return model

    logging.debug('Predicting on test with model')

    fold_model_names = [cv_results[fold]['model_path'] for fold in cv_results if fold != 'summary']

    all_probas = []
    for model_path in fold_model_names:
        model = LeafDoctorModel(hparams = Namespace(**parameters['classifier']))
        model.load_state_dict(torch.load(model_path))
        finetune_cv_model(model, train, parameters)

        predictions, probas = predict(model,
                                  dataset=test,
                                  indices=list(range(len(test))),
                                  batch_size=parameters['eval']['batch_size'],
                                  num_workers=parameters['data_loader_workers'],
                                  transform=get_test_transforms(parameters['classifier']['test_width'], parameters['classifier']['test_height']))

        all_probas.append(probas)

    aggregated_probas = np.mean(all_probas, axis=0).reshape(-1, 5)
    pred_labels = np.argmax(aggregated_probas, 1)
    sample_submission.label = pred_labels
    return sample_submission
