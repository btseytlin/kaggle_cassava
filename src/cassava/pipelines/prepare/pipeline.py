from kedro.pipeline import Pipeline, node

from .nodes import prepare_dataset, find_duplicates, obtain_image_hashes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                obtain_image_hashes,
                ['train_images_torch_2020', 'train_images_torch_2019', 'test_images_torch_2019', 'extra_images_torch_2019'],
                ['image_ids', 'image_hashes'],
                name='obtain_image_hashes'
            ),
            node(
                find_duplicates,
                ['image_ids', 'image_hashes'],
                'duplicates',
                name='find_duplicates'
            ),
            node(
                prepare_dataset,
                ['train_images_torch_2020', 'train_images_torch_2019', 'test_images_torch_2019', 'extra_images_torch_2019', 'duplicates'],
                ["train", "unlabelled"],
                name="prepare_dataset"
            ),
        ]
    )
