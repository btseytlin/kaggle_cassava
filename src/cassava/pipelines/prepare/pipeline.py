from kedro.pipeline import Pipeline, node

from .nodes import prepare_lmdb


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                prepare_lmdb,
                ["train_images_torch", "test_images_torch"],
                None,
                name="prepare_lmdb"
            ),
        ]
    )
