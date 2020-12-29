import logging
from collections import defaultdict
from copy import copy
import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import StratifiedKFold

from cassava.pipelines.finetune.nodes import finetune_byol_test, finetune_classifier_resolution
from cassava.pipelines.pretrain.nodes import pretrain_model
from cassava.pipelines.train_model.nodes import train_model, score_model


from cassava.utils import DatasetFromSubset
from torch.utils.data import Subset


def obtain_cv_splits(train, parameters):
    labels = train.labels
    sources = train.sources
    indices_2020 = np.argwhere(sources == 'train_2020').flatten()
    indices_2019 = np.argwhere(sources == 'train_2019').flatten()

    cv = StratifiedKFold(n_splits=parameters['cv_splits'], random_state=parameters['seed'])

    splits = []
    # Preserve same class distribution in both train and test
    # Only put 2020 data in test
    splits_2019 = list(cv.split(indices_2019, labels[indices_2020][:len(indices_2019)]))
    splits_2020 = list(cv.split(indices_2020, labels[indices_2020]))
    for (train_2019_idx, test_2019_idx), (train_2020_idx, test_2020_idx) in zip(splits_2019, splits_2020):
        train_idx = np.concatenate([indices_2019[train_2019_idx], indices_2020[train_2020_idx]])
        test_idx = indices_2020[test_2020_idx]
        splits.append((train_idx, test_idx))
    return splits


def cross_validation(train, unlabelled, cv_splits, parameters):
    cv_results = {
        'summary': {},
    }
    score_values = {
        'test': defaultdict(list),
        'val': defaultdict(list),
    }

    if os.path.exists(parameters['cv_models_dir']) and len(os.listdir(parameters['cv_models_dir'])) > 0:
        raise Exception('CV models path already exists, please delete it explicitly to overwrite')
    else:
        os.makedirs(parameters['cv_models_dir'], exist_ok=True)

    for fold_num, (train_idx, test_idx) in enumerate(cv_splits):
        logging.info('Fitting CV fold %d', fold_num)
        model_path = os.path.join(parameters['cv_models_dir'], f'model_fold_{fold_num}.pt')
        fold_parameters = copy(parameters)

        fold_train_dataset = DatasetFromSubset(Subset(train, indices=train_idx))
        fold_test_dataset = DatasetFromSubset(Subset(train, indices=test_idx))

        # Split
        logging.info('Pretraining on train+unlabelled')
        pretrained_model = pretrain_model(fold_train_dataset, unlabelled, fold_parameters)

        logging.info('Training on train')
        model = train_model(pretrained_model, fold_train_dataset, fold_parameters)

        logging.info('Finetuning with BYOL')
        model = finetune_byol_test(model, fold_train_dataset, fold_test_dataset, fold_parameters)

        logging.info('Finetuning for test resolution')
        model = finetune_classifier_resolution(model, fold_train_dataset, fold_parameters)

        logging.info('Done training CV fold')
        torch.save(model.state_dict(), model_path)

        # Score on test
        test_scores, test_predictions = score_model(model, fold_test_dataset, list(range(len(fold_test_dataset))), fold_parameters)

        cv_results[f'fold_{fold_num}'] = {
            'model_path': model_path,
            'test_indices': test_idx,
            'test_scores': test_scores,
            'test_predictions': test_predictions,
        }

        for score in test_scores:
            score_values['test'][score].append(test_scores[score])

    for score_set in score_values:
        for score_name, scores in score_values[score_set].items():
            cv_results['summary'][f'{score_set}_{score_name}_mean'] = np.mean(scores)
            cv_results['summary'][f'{score_set}_{score_name}_std'] = np.std(scores)

    logging.info('Cross-validation results %s', cv_results['summary'])
    return cv_results
