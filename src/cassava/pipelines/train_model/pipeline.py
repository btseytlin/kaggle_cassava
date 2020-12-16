from kedro.pipeline import Pipeline, node

from .nodes import split_data, train_model, score_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                ["train_labels", "parameters"],
                ["train_indices", "val_indices"],
                name='split_data',
            ),
            node(
                train_model,
                ["pretrained_model", "train_images_lmdb", "train_indices", "val_indices", "parameters"],
                "model",
                name='train_model',
            ),
            node(
                score_model,
                ["model", "train_images_lmdb", "val_indices", "parameters"],
                ["val_scores", "val_predictions"],
                name='val_score',
            ),
        ]
    )
