from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal

from cassava.pipelines import prepare, pretrain, train_model, cross_validation, predict, finetune


class ProjectHooks:
    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """
        prepare_pipeline = prepare.create_pipeline()
        pretrain_pipeline = pretrain.create_pipeline()
        finetune_pipeline = finetune.create_pipeline()
        train_pipeline = train_model.create_pipeline()
        predict_pipeline = predict.create_pipeline()
        cv_pipeline = cross_validation.create_pipeline()

        return {
            "prepare": prepare_pipeline,
            "pretrain": pretrain_pipeline,
            "train": train_pipeline,
            "predict": predict_pipeline,
            "cv": cv_pipeline,
            "finetune": finetune_pipeline,
            "__submit__": prepare_pipeline + finetune_pipeline + cv_pipeline + predict_pipeline,
            "__default__": cv_pipeline + predict_pipeline,
        }

    @hook_impl
    def register_config_loader(self, conf_paths: Iterable[str]) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


project_hooks = ProjectHooks()
