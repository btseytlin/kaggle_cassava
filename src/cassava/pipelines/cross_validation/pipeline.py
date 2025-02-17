from kedro.pipeline import Pipeline, node

from .nodes import cross_validation, obtain_cv_splits


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                obtain_cv_splits,
                ["train", "parameters"],
                "cv_splits"
            ),
            node(
                cross_validation,
                ["train", "unlabelled", "cv_splits", "parameters"],
                "cv_results",
                name='cross_validation',
            ),
        ]
    )
