from kedro.pipeline import Pipeline, node

from .nodes import finetune_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                finetune_model,
                ["model", "train_images_lmdb", "test_images_lmdb", "parameters"],
                "finetuned_model",
                name="finetune"
            ),
        ]
    )
