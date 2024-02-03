from typing import Any
from datasets import Dataset
from kedro.io import AbstractDataset


class HFDataset(AbstractDataset):
    def __init__(self, file_path):
        self._file_path = file_path

    def _load(self) -> Dataset:
        return Dataset.load_from_disk(self._file_path)

    def _save(self, data: Dataset) -> None:
        data.save_to_disk(self._file_path)

    def _describe(self) -> dict[str, Any]:
        return {
            "file_path": self._file_path
        }
