from kedro.pipeline import Pipeline, node

from .nodes import finetune_on_test


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                finetune_on_test,
                ["pretrained_model", "train_images_lmdb", "test_images_torch_2020", "parameters"],
                "finetuned_model",
                name="finetune_on_test"
            ),
        ]
    )
