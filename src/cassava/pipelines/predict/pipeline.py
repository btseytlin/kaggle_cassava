from kedro.pipeline import Pipeline, node

from .nodes import predict_submission


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                predict_submission,
                ["cv_results", "test_images_torch", "sample_submission", "parameters"],
                "submission",
                name='predict',
            ),
        ]
    )
