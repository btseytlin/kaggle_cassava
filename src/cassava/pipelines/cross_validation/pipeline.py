from kedro.pipeline import Pipeline, node

from .nodes import cross_validation


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cross_validation,
                ["pretrained_model", "train_images_torch", "test_images_torch", "parameters"],
                "cv_results",
                name='cross_validation',
            ),
        ]
    )
