import os
import pandas as pd
from PIL import Image
from skimage import io
from kedro.io import AbstractDataSet
import copy
from pathlib import Path
from typing import Any, Dict

import logging

from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from cassava.utils import CassavaDataset

log = logging.getLogger(__name__)


class ImageOneFolderDataSet(AbstractDataSet):
    """Loads a folder containing images as an iterable.
    """

    def __init__(
        self,
        filepath: str,
        labels_path: str,
        images_path: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        """
        Args:
            filepath:
            load_args:
            save_args: Ignored as saving is not supported.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded.
        """

        super().__init__()
        self._filepath = filepath
        self._labels_path = os.path.join(self._filepath, labels_path)
        self._images_path = os.path.join(self._filepath, images_path)
        self._load_args = load_args
        self._save_args = save_args

    def _load(self) -> Any:
        load_path = Path(self._labels_path)
        load_args = copy.deepcopy(self._load_args)
        load_args = load_args or dict()
        df = pd.read_csv(load_path)
        load_args['root'] = Path(self._images_path)
        dataset = CassavaDataset(
            image_ids=df['image_id'].values,
            labels=df['label'].values,
            **load_args
        )

        return dataset

    def _save(self, vision_dataset) -> None:
        """ Not Implemented """
        raise NotImplementedError()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._save_args,
            save_args=self._save_args,
        )


class ImageFolderDataSet(AbstractDataSet):
    """Wrapper over torch ImageFolder dataset
    """

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        """
        Args:
            filepath:
            load_args:
            save_args: Ignored as saving is not supported.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded.
        """

        super().__init__()
        self._filepath = filepath
        self._load_args = load_args
        self._save_args = save_args

    def _load(self) -> Any:
        load_path = Path(self._filepath)
        load_args = copy.deepcopy(self._load_args)
        load_args = load_args or dict()
        dataset = ImageFolder(
            root=load_path,
            **load_args
        )
        return dataset

    def _save(self, vision_dataset) -> None:
        """ Not Implemented """
        raise NotImplementedError()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._save_args,
            save_args=self._save_args,
        )

