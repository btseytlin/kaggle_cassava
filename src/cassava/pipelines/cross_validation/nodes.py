import logging
from copy import copy

import numpy as np
import os
import torch
from sklearn.model_selection import StratifiedKFold
from cassava.pipelines.train_model.nodes import train_model, score_model


def cross_validation(pretrained_model, train_images_lmdb, test_images_lmdb, parameters):
    cv_results = {}
    score_values = {}

    if os.path.exists(parameters['cv_models_dir']):
        raise Exception('CV models path already exists, please delete it explicitly to overwrite')
    else:
        os.makedirs(parameters['cv_models_dir'])

    cv = StratifiedKFold(n_splits=parameters['cv_splits'], random_state=parameters['seed'])
    indices = np.array(list(range(len(train_images_lmdb))))
    labels = train_images_lmdb.labels
    for fold_num, (train_idx, val_idx) in enumerate(cv.split(indices, labels)):
        logging.info('Fitting CV fold %d', fold_num)
        model_path = os.path.join(parameters['cv_models_dir'], f'model_fold_{fold_num}.pt')
        fold_parameters = copy(parameters)
        model = train_model(pretrained_model, train_images_lmdb, train_idx, val_idx, fold_parameters)
        torch.save(model.state_dict(), model_path)
        scores, oof_predictions = score_model(model, train_images_lmdb, val_idx, fold_parameters)
        cv_results[f'fold_{fold_num}'] = {
            'model_path': model_path,
            'scores': scores,
            'val_indices': val_idx,
            'oof_predictions': oof_predictions,
        }

        for score in scores:
            if not score_values.get(score):
                score_values[score] = []
            score_values[score].append(scores[score])

    cv_results['summary'] = {}
    for score_name, scores in score_values.items():
        cv_results['summary'][f'{score_name}_mean'] = np.mean(scores)
        cv_results['summary'][f'{score_name}_std'] = np.std(scores)

    logging.info('Cross-validation results %s')
    return cv_results
