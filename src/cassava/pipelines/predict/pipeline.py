from kedro.pipeline import Pipeline, node

from .nodes import predict_submission


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                predict_submission,
                ["finetuned_model", "test_images_lmdb", "sample_submission", "parameters"],
                "submission",
                name='predict',
            ),
        ]
    )
