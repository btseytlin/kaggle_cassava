from kedro.pipeline import Pipeline, node

from .nodes import predict_submission, prepare_test_dataset


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                prepare_test_dataset,
                ["test_images_torch_2020"],
                "test",
                name='prepare_test_dataset',
            ),
            node(
                predict_submission,
                ["cv_results", "train", "test", "sample_submission", "parameters"],
                "submission",
                name='predict',
            ),
        ]
    )
