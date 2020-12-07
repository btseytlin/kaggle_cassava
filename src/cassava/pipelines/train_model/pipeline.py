from kedro.pipeline import Pipeline, node

from .nodes import split_data, train_model, score_model, report_on_training


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                ["train_labels", "parameters"],
                ["train_indices", "val_indices"],
                name='split',
            ),
            node(
                train_model,
                ["train_images_torch", "train_indices", "val_indices", "parameters"],
                "model",
                name='train',
            ),
            node(
                score_model,
                ["model", "train_images_torch", "val_indices", "parameters"],
                "val_scores",
                name='val_score',
            ),
        ]
    )
