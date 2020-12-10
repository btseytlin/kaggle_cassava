from kedro.io import AbstractDataSet
import copy
from pathlib import Path
from typing import Any, Dict

import logging


from cassava.lmdb_dataset import ImageLMDBDataset


log = logging.getLogger(__name__)


class KedroImageLMDBDataSet(AbstractDataSet):
    """Loads an lmdb file as a torch dataset.
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
        dataset = ImageLMDBDataset(
            db_path=load_path,
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

