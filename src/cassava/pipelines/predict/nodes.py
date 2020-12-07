import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from cassava.transforms import get_train_transforms, get_test_transforms


def predict(model, test_images_torch, sample_submission, parameters):
    logging.debug('Predicting with model')

    test_images_torch.transform = get_test_transforms()
    loader = torch.utils.data.DataLoader(test_images_torch, batch_size=parameters['batch_size'])

    predictions = []
    model.eval()
    model.freeze()
    for images, labels in tqdm(loader):
        batch_preds = model.predict(images)
        predictions += batch_preds.tolist()

    sample_submission.label = predictions

    return sample_submission
