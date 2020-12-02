from typing import Any, Dict

import pandas as pd
import os

def make_image_folder(train, train_images):
    image_folder_dir = os.path.join('data', '03_primary', 'train_images')
    for ix, row in train.iterrows():
        label = str(row.label)
        image_id = row.image_id

        image_key = image_id.split('.')[0]



        image_dataset = train_images[image_key]

        os.makedirs(os.path.join(image_folder_dir, label), exist_ok=True)

        image = image_dataset()

        image.save(os.path.join(image_folder_dir, label, image_id), 'JPEG')

