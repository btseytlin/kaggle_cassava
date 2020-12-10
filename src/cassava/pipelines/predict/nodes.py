import logging
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cassava.transforms import get_train_transforms, get_test_transforms
from cassava.utils import DatasetFromSubset
from cassava.models.model import LeafDoctorModel
from cassava.node_helpers import predict


def predict_submission(cv_results, test_images_torch, sample_submission, parameters):
    logging.debug('Predicting on test with model')

    fold_model_names = [cv_results[fold]['model_path'] for fold in cv_results if fold != 'summary']

    all_probas = []
    for model_path in fold_model_names:
        model = LeafDoctorModel()
        model.load_state_dict(torch.load(model_path))

        predictions, probas = predict(model,
                                  dataset=test_images_torch,
                                  indices=list(range(len(test_images_torch))),
                                  batch_size=parameters['classifier']['batch_size'],
                                  num_workers=parameters['data_loader_workers'],
                                  transform=get_test_transforms())

        all_probas.append(probas)

    aggregated_probas = np.mean(all_probas, axis=0)
    pred_labels = np.argmax(aggregated_probas, 1)
    sample_submission.label = pred_labels
    return sample_submission
