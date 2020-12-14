import numpy as np
import pandas as pd
import seaborn as sns
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


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


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
