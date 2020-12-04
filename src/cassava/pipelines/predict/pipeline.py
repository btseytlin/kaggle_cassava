from kedro.pipeline import Pipeline, node

from .nodes import predict


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                predict,
                ["model", "test_images_torch", "sample_submission", "parameters"],
                "submission",
                name='predict',
            ),
        ]
    )
