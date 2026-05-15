from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.path_vqa_datasets import PathVQADataset, PathVQAEvalDataset


@registry.register_builder("path_vqa")
class PathVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PathVQADataset
    eval_dataset_cls = PathVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/pathvqa/defaults.yaml"}

    def build(self):
        datasets = super().build()
        prompt_mode = self.config.get("prompt_mode", "typed")

        for dataset in datasets.values():
            dataset.prompt_mode = prompt_mode

        return datasets

    def _download_ann(self):
        # PathVQA is expected to be prepared locally; do not try to download/copy
        # pickle annotations through torchvision's download helpers.
        return
