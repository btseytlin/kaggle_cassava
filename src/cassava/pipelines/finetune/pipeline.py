from kedro.pipeline import Pipeline, node

from .nodes import finetune_byol_test, finetune_classifier_resolution


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                finetune_byol_test,
                ["pretrained_model", "train", "test_images_torch_2020", "parameters"],
                "finetuned_byol_model",
                name="finetune_byol_test"
            ),
            node(
                finetune_classifier_resolution,
                ["finetuned_byol_model", "train", "parameters"],
                "finetuned_model",
                name="finetune_classifier_resolution"
            ),
        ]
    )
