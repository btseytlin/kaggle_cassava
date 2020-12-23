from kedro.pipeline import Pipeline, node

from .nodes import pretrain_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                pretrain_model,
                ["train_lmdb", "unlabelled_lmdb", "parameters"],
                "pretrained_model",
                name="pretrain"
            ),
        ]
    )
