from kedro.pipeline import Pipeline, node

from .nodes import cross_validation


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cross_validation,
                ["finetuned_model", "train_images_lmdb", "parameters"],
                "cv_results",
                name='cross_validation',
            ),
        ]
    )
