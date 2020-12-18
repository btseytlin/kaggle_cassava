import logging
from collections import defaultdict
from copy import copy

import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import StratifiedKFold

from cassava.pipelines.finetune.nodes import finetune_on_test
from cassava.pipelines.pretrain.nodes import pretrain_model
from cassava.pipelines.train_model.nodes import train_model, score_model, split_data
from cassava.utils import DatasetFromSubset
from torch.utils.data import Subset


def cross_validation(train_images_lmdb, parameters):
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

    cv = StratifiedKFold(n_splits=parameters['cv_splits'], random_state=parameters['seed'])
    indices = np.array(list(range(len(train_images_lmdb))))
    labels = train_images_lmdb.labels
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(indices, labels)):
        logging.info('Fitting CV fold %d', fold_num)

        model_path = os.path.join(parameters['cv_models_dir'], f'model_fold_{fold_num}.pt')
        fold_parameters = copy(parameters)
        fold_train_dataset = DatasetFromSubset(Subset(train_images_lmdb, indices=train_idx))
        fold_test_dataset = DatasetFromSubset(Subset(train_images_lmdb, indices=test_idx))
        train_labels = fold_train_dataset.labels
        train_labels_df = pd.DataFrame({'label': train_labels}, index=range(len(train_labels)))

        logging.info('Pretraining on train+val')
        pretrained_model = pretrain_model(fold_train_dataset, fold_parameters)

        logging.info('Finetuning on train+val+test')
        finetuned_model = finetune_on_test(pretrained_model, fold_train_dataset, fold_test_dataset, fold_parameters)

        # Split
        fold_train_idx, fold_val_idx = split_data(train_labels_df, fold_parameters)
        global_val_idx = train_idx[fold_val_idx]
        global_train_idx = train_idx[fold_train_idx]

        # Assert no leakage of test into train
        assert not bool(set(global_train_idx).intersection(set(global_val_idx)))
        assert not bool(set(global_train_idx).union(set(global_val_idx)).intersection(set(test_idx)))

        logging.info('Training on train, early stopping using val')
        model = train_model(finetuned_model, fold_train_dataset, fold_train_idx, fold_val_idx, fold_parameters)

        torch.save(model.state_dict(), model_path)

        # Score on validation
        val_scores, oof_predictions = score_model(model, fold_train_dataset, fold_val_idx, fold_parameters)

        # Score on test
        test_scores, test_predictions = score_model(model, fold_test_dataset, list(range(len(fold_test_dataset))), fold_parameters)

        cv_results[f'fold_{fold_num}'] = {
            'model_path': model_path,
            'val_indices': global_val_idx,
            'test_indices': test_idx,
            'val_scores': val_scores,
            'oof_predictions': oof_predictions,
            'test_scores': test_scores,
            'test_predictions': test_predictions,
        }

        for score in test_scores:
            score_values['test'][score].append(test_scores[score])
            score_values['val'][score].append(val_scores[score])

    for score_set in score_values:
        for score_name, scores in score_values[score_set].items():
            cv_results['summary'][f'{score_set}_{score_name}_mean'] = np.mean(scores)
            cv_results['summary'][f'{score_set}_{score_name}_std'] = np.std(scores)

    logging.info('Cross-validation results %s', cv_results['summary'])
    return cv_results
