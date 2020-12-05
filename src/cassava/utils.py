import numpy as np
import pandas as pd
import os
from skimage import io
from torch.utils.data import Dataset
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


class Unnormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def plot_image(img, label=None, ax=None):
    img = torch.Tensor(np.array(img))
    label_num_to_disease_map = {0: 'Cassava Bacterial Blight (CBB)',
                                1: 'Cassava Brown Streak Disease (CBSD)',
                                2: 'Cassava Green Mottle (CGM)',
                                3: 'Cassava Mosaic Disease (CMD)',
                                4: 'Healthy'}

    if not ax:
        ax = plt.gca()
    ax.imshow(img.permute(2, 1, 0))
    ax.axis('off')
    if label is not None:

        if isinstance(label, int):
            label = label_num_to_disease_map.get(label, 0)
        ax.set_title(f'{label}')


def plot_label_examples(dataset, targets, target_label):
    label_indices = np.where(targets == target_label)[0]

    sample = np.random.choice(label_indices, 6)

    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, idx in zip(grid, sample):
        img, label = dataset[idx]
        assert label == target_label
        plot_image(img, ax=ax)
    plt.suptitle(f'Label {target_label}')
    plt.show()


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(image=x)['image']
        return x, y

    def __len__(self):
        return len(self.subset)


class CassavaDataset(Dataset):
    def __init__(self, root, image_ids, labels, transform=None):
        super().__init__()
        self.root = root
        self.image_ids = image_ids
        self.labels = labels
        self.targets = self.labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = io.imread(os.path.join(self.root, self.image_ids[idx]))

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label
