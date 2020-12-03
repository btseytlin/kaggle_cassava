from kedro.pipeline import Pipeline, node

from .nodes import make_image_folder


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                make_image_folder,
                ['train_labels', 'train_images'],
                None,
                name='make_image_folder'
            )
        ]
    )

