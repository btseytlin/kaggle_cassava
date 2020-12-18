import logging
from collections import defaultdict
from copy import copy

import numpy as np
import os
import torch
from sklearn.model_selection import StratifiedKFold

from cassava.pipelines.finetune.nodes import finetune_on_test
from cassava.pipelines.pretrain.nodes import pretrain_model
from cassava.pipelines.train_model.nodes import train_model, score_model, split_data
from cassava.utils import DatasetFromSubset


def cross_validation(train_images_lmdb, parameters):
    cv_results = {}
    score_values = {
        'test': defaultdict(list),
        'val': defaultdict(list),
    }

    if os.path.exists(parameters['cv_models_dir']):
        raise Exception('CV models path already exists, please delete it explicitly to overwrite')
    else:
        os.makedirs(parameters['cv_models_dir'])

    cv = StratifiedKFold(n_splits=parameters['cv_splits'], random_state=parameters['seed'])
    indices = np.array(list(range(len(train_images_lmdb))))
    labels = train_images_lmdb.labels
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(indices, labels)):
        logging.info('Fitting CV fold %d', fold_num)

        model_path = os.path.join(parameters['cv_models_dir'], f'model_fold_{fold_num}.pt')
        fold_parameters = copy(parameters)
        fold_train_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_lmdb, indices=train_idx))
        fold_test_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_lmdb, indices=test_idx))
        train_labels = labels[train_idx]

        # Pretrain on train+val
        pretrained_model = pretrain_model(fold_train_dataset, fold_parameters)

        # Finetune on test
        finetuned_model = finetune_on_test(pretrained_model, fold_train_dataset, fold_test_dataset, fold_parameters)

        # Split
        train_idx, val_idx = split_data(train_labels, fold_parameters)

        # Assert no leakage of test into train
        assert not bool(train_idx.intersection(set(val_idx)))
        assert not bool(set(train_idx).union(set(val_idx)).intersection(set(test_idx)))

        # Train
        model = train_model(finetuned_model, fold_train_dataset, train_idx, val_idx, fold_parameters)

        torch.save(model.state_dict(), model_path)

        # Score on validation
        val_scores, oof_predictions = score_model(model, fold_train_dataset, val_idx, fold_parameters)

        # Score on test
        test_scores, test_predictions = score_model(model, fold_test_dataset, test_idx, fold_parameters)

        cv_results[f'fold_{fold_num}'] = {
            'model_path': model_path,
            'val_indices': val_idx,
            'test_indices': test_idx,
            'val_scores': val_scores,
            'oof_predictions': oof_predictions,
            'test_scores': test_scores,
            'test_predictions': test_predictions,
        }

        for score in test_scores:
            score_values['test'][score].append(test_scores[score])
            score_values['val'][score].append(val_scores[score])

    cv_results['summary'] = {}
    for score_set in score_values:
        for score_name, scores in score_values[score_set].items():
            cv_results['summary'][f'{score_set}_{score_name}_mean'] = np.mean(scores)
            cv_results['summary'][f'{score_set}_{score_name}_std'] = np.std(scores)

    logging.info('Cross-validation results %s')
    return cv_results
