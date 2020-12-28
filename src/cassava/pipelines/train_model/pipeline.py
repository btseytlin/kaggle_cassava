from kedro.pipeline import Pipeline, node

from .nodes import split_data, train_model, score_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["pretrained_model", "train", "parameters"],
                "model",
                name='train',
            ),
        ]
    )
