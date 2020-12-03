from kedro.pipeline import Pipeline, node

from .nodes import split_data, train_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                ["train_labels", "parameters"],
                ["train_indices", "val_indices"],
            ),
            node(
                train_model,
                ["train_images_torch", "train_indices", "val_indices", "parameters"],
                ["model", "metrics_history"],
            ),
        ]
    )
