from typing import Callable, Literal

import torch

from pixeltable.catalog import Table
from torch.utils.data import Dataset

from goldener.extract import GoldFeatureExtractor
from goldener.pxt_utils import create_pxt_table_from_sample


class GoldDescriptor:
    """Describe the `data` of a dataset by extracting features and storing them in a PixelTable table.

    Assuming all the data will not fit in memory, the dataset is processed in batches.
    All the data of the dataset, the computed features included, will be saved in a local Pixeltable table.
    The `data` of the dataset will be saved in the shape and scale obtained after applying the `collate_fn` if provided.
    These arrays are expected to be all the same size. All torch tensors will be converted to numpy arrays before saving.

    Attributes:
        table_path: Path to the PixelTable table where the description will be saved locally.
        extractor: FeatureExtractor instance for feature extraction.
        collate_fn: Optional function to collate dataset samples into batches composed of
        dictionaries with at least the `data` key returning a pytorch Tensor.
        If None, the dataset is expected to directly provide such batches. It should as well format the `data` value
        in the format expected by the feature extractor.
        batch_size: Optional batch size for processing the dataset.
        num_workers: Optional number of worker threads for data loading.
        if_exists: Behavior if the table already exists ('error' or 'replace_force'). If 'replace_force',
        the existing table will be replaced, otherwise an error will be raised.
        distribute: Whether to use distributed processing for feature extraction and table population. Not implemented yet.
    """

    def __init__(
        self,
        table_path: str,
        extractor: GoldFeatureExtractor,
        collate_fn: Callable | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        if_exists: Literal["error", "replace_force"] = "error",
        distribute: bool = False,
        device: torch.device | None = None,
    ):
        self.table_path = table_path
        self.extractor = extractor
        self.collate_fn = collate_fn
        self.if_exists = if_exists
        self.distribute = distribute

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.batch_size: int | None
        self.num_workers: int | None
        if not self.distribute:
            self.batch_size = 1 if batch_size is None else batch_size
            self.num_workers = 0 if num_workers is None else num_workers
        else:
            self.batch_size = batch_size
            self.num_workers = num_workers

    def describe(self, dataset: Dataset) -> Table:
        """Describe the dataset by extracting features and storing them in a PixelTable table.

        Args:
            dataset: Dataset to be described. Each item should be a dictionary with at least the `data` key
            after applying the collate_fn. If the collate_fn is None, the dataset is expected to directly
            provide such batches.
        """
        sample = dataset[0]
        if self.collate_fn is not None:
            sample = self.collate_fn([sample])

        if not isinstance(sample, dict):
            raise ValueError(
                "Dataset items must be dictionaries after applying the collate_fn."
            )

        if "data" not in sample:
            raise ValueError("Dataset items must contain a 'data' key.")

        pxt_table = self._initialize_table(sample)

        if self.distribute:
            self._distributed_describe(pxt_table, dataset)
        else:
            pxt_table = self._sequential_describe(pxt_table, dataset)

        return pxt_table

    def _initialize_table(self, sample: dict) -> Table:
        """Initialize the PixelTable table using a single sample from the dataset.

        It creates the required PixelTable directories and table schema based on the provided sample.

        Args:
            sample: A single sample from the dataset to initialize the table schema.
            It should be a dictionary with at least the `data` key. The  `data` value must be formatted
            in the format expected by the feature extractor.
        """
        sample["features"] = self.extractor.extract_and_fuse(
            sample["data"].to(device=self.device)
        )
        if self.collate_fn is not None:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    # Only initial tensors will have a batch dimension added by the collate_fn
                    # otherwise, they were initially single values (float or int)
                    if value.ndim > 1:
                        sample[key] = value.squeeze(0)
                    else:
                        sample[key] = value.item()
                else:
                    # non tensor values are expected to be lists of single values
                    sample[key] = value[0]

        if "idx" not in sample:
            sample["idx"] = 0

        pxt_table = create_pxt_table_from_sample(
            self.table_path, sample, self.if_exists
        )
        pxt_table.where(pxt_table.idx == 0).delete()  # remove the initial sample

        return pxt_table

    def _distributed_describe(
        self,
        pxt_table: Table,
        dataset: Dataset,
    ) -> Table:
        raise NotImplementedError("Distributed description is not implemented yet.")

    def _sequential_describe(
        self,
        pxt_table: Table,
        dataset: Dataset,
    ) -> Table:
        assert self.batch_size is not None
        assert self.num_workers is not None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.num_workers is not None else 0,
            collate_fn=self.collate_fn,
        )
        for batch in dataloader:
            batch["features"] = self.extractor.extract_and_fuse(
                batch["data"].to(device=self.device)
            )
            if "idx" not in batch:
                batch["idx"] = [idx for idx in range(len(batch["data"]))]
            pxt_table.insert(
                [
                    {
                        key: (
                            (
                                value[sample_idx].item()
                                if value.ndim == 1
                                else value[sample_idx].detach().cpu().numpy()
                            )
                            if isinstance(value, torch.Tensor)
                            else value[sample_idx]
                        )
                        for key, value in batch.items()
                    }
                    for sample_idx in range(len(batch["data"]))
                ]
            )

        return pxt_table
