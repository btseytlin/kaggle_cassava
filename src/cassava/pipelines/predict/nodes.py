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


def predict_submission(model, test_images_lmdb, sample_submission, parameters):
    logging.debug('Predicting with model')
    predictions, probas = predict(model,
                                  dataset=test_images_lmdb,
                                  indices=list(range(len(test_images_lmdb))),
                                  batch_size=parameters['classifier']['batch_size'],
                                  num_workers=parameters['data_loader_workers'])

    sample_submission.label = predictions

    return sample_submission
