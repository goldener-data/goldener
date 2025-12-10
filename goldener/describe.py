from typing import Callable

import torch

import pixeltable as pxt
from pixeltable import Error
from pixeltable.catalog import Table
from torch.utils.data import Dataset, DataLoader

from goldener.extract import GoldFeatureExtractor
from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    get_expr_from_column_name,
    get_sample_row_from_idx,
    pxt_torch_dataset_collate_fn,
    get_valid_table,
    include_batch_into_table,
)
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import filter_batch_from_indices


class GoldDescriptor:
    """Describe the `data` of a dataset by extracting features from a pretrained model.

    Assuming all the data will not fit in memory, the dataset is processed in batches.

    All the data of the dataset, the computed features included, will be saved in a local Pixeltable table during
    the computation. In this table, the `data` of the dataset will be saved in the shape and scale obtained
    after applying the `collate_fn` if provided. These arrays are expected to be all the same size.
    All torch tensors will be converted to numpy arrays before saving.

    The description can operate in a sequential (single-process) mode or a
    distributed mode (not implemented). The table schema is created/validated automatically and will include
    minimal indexing columns (`idx`) required to link description back to
    their originating samples.

    Attributes:
        table_path: Path to the PixelTable table where the description will be saved locally.
        extractor: FeatureExtractor instance for feature extraction.
        transform: Transformation to apply to the `data` before running the feature extraction if it is not already
        applied by the `collate_fn`.
        collate_fn: Optional function to collate dataset samples into batches composed of
        dictionaries with at least the key specified by `data_key` returning a pytorch Tensor.
        If None, the dataset is expected to directly provide such batches. It should as well format
        the value at `data_key` in the format expected by the feature extractor.
        data_key: Key in the batch dictionary that contains the data to extract features from. Default is "data".
        description_key: Key in the table where the extracted features will be stored. Default is "features".
        to_keep_schema: Optional dictionary defining additional columns to keep from the original dataset/table
        into the description table. The keys are the column names and the values are the PixelTable types.
        batch_size: Optional batch size for processing the dataset.
        num_workers: Optional number of worker threads for data loading.
        allow_existing: If False, an error will be raised when the table already exists.
        distribute: Whether to use distributed processing for feature extraction and table population. Not implemented yet.
        drop_table: Whether to drop the description table after creating the dataset with descriptions. It is only applied
        when using `describe_in_dataset`.
        device: Torch device to use for feature extraction. If None, it will use 'cuda' if available, otherwise 'cpu'.
        max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Int,
    }

    def __init__(
        self,
        table_path: str,
        extractor: GoldFeatureExtractor,
        transform: Callable | None = None,
        collate_fn: Callable | None = None,
        data_key: str = "data",
        description_key: str = "features",
        to_keep_schema: dict[str, type] | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        device: torch.device | None = None,
        max_batches: int | None = None,
    ):
        self.table_path = table_path
        self.extractor = extractor
        self.transform = transform
        self.collate_fn = collate_fn
        self.data_key = data_key
        self.description_key = description_key
        self.to_keep_schema = to_keep_schema
        self.allow_existing = allow_existing
        self.distribute = distribute
        self.drop_table = drop_table
        self.max_batches = max_batches

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

    def describe_in_dataset(
        self,
        to_describe: Dataset | Table,
    ) -> GoldPxtTorchDataset:
        """Describe the data by extracting features and returning them within a new dataset.

        The description process extracts features from the data using the provided feature extractor
        and stores them in a PixelTable table specified by `table_path`. The description of each data
        will be stored in the column specified by `description_key`.

        This is a convenience wrapper that runs `describe_in_table` to populate
        (or resume populating) the PixelTable table, then wraps the table into a
        `GoldPxtTorchDataset` for downstream consumption. If `drop_table` is True,
        the table will be removed after the dataset is created.

        Args:
            to_describe: Dataset or Table to be described. If a Dataset is provided, each item should be a
            dictionary with at least the key specified by `data_key` after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
            it should contain both 'idx' and `data_key` column.

        Returns:
            A GoldPxtTorchDataset dataset containing at least the extracted features in the `description_key` key
            and an `idx` key as well.
        """

        description_table = self.describe_in_table(to_describe)

        description_dataset = GoldPxtTorchDataset(description_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(description_table)

        return description_dataset

    def describe_in_table(
        self,
        to_describe: Dataset | Table,
    ) -> Table:
        """Describe the data by extracting features and storing them in a PixelTable table.

        The description process extracts features from the data using the provided feature extractor
        and stores them in a PixelTable table specified by `table_path`. The description of each data
        will be stored in the column specified by `description_key`.

        This method is idempotent (i.e. failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will not duplicate or recompute the descriptions
        already present in the PixelTable table.

        Args:
            to_describe: Dataset or Table to be described. If a Dataset is provided, each item should be a
            dictionary with at least the key specified by `data_key` after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
            it should contain both 'idx' and `data_key` column.

        Returns:
            A PixelTable Table containing at least the extracted features in the `description_key` column
            and an `idx` column as well.
        """
        # If the computation was already started or already done, we resume from there
        try:
            old_description_table = pxt.get_table(
                self.table_path,
                if_not_exists="ignore",
            )
        except Error:
            old_description_table = None

        if not self.allow_existing and old_description_table is not None:
            raise ValueError(
                f"Table at path {self.table_path} already exists and "
                "allow_existing is set to False."
            )

        # get the table and dataset to execute the description pipeline
        to_describe_dataset: GoldPxtTorchDataset | Dataset
        if isinstance(to_describe, Table):
            description_table = self._description_table_from_table(
                to_describe, old_description_table
            )

            if description_table.count() > 0 and "idx" in to_describe.columns():
                to_describe_indices = set(
                    [
                        row["idx"]
                        for row in to_describe.select(to_describe.idx).collect()
                    ]
                )
                already_described = set(
                    [
                        row["idx"]
                        for row in description_table.select(
                            description_table.idx
                        ).collect()
                    ]
                )
                if not to_describe_indices.difference(already_described):
                    return description_table

            to_describe_dataset = GoldPxtTorchDataset(to_describe)

        else:
            to_describe_dataset = to_describe
            description_table = self._description_table_from_dataset(
                to_describe_dataset, old_description_table
            )

        if self.distribute:
            described = self._distributed_describe(
                description_table, to_describe_dataset
            )
        else:
            described = self._sequential_describe(
                description_table, to_describe_dataset
            )

        return described

    def _description_table_from_table(
        self, to_describe: Table, old_description_table: Table | None
    ) -> Table:
        minimal_schema = self._MINIMAL_SCHEMA
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        description_table = get_valid_table(
            table=old_description_table
            if old_description_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
        )

        if self.description_key not in description_table.columns():
            sample = get_sample_row_from_idx(
                to_describe,
                collate_fn=pxt_torch_dataset_collate_fn,
                expected_keys=[self.data_key],
            )
            sample_data = sample[self.data_key]
            if self.transform is not None:
                sample_data = self.transform(sample_data)

            description = (
                self.extractor.extract_and_fuse(sample_data.to(device=self.device))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            description_table.add_column(
                **{
                    self.description_key: pxt.Array[  # type: ignore[misc]
                        description.shape, pxt.Float
                    ]
                }
            )

        return description_table

    def _description_table_from_dataset(
        self, to_describe: Dataset, old_description_table: Table | None
    ) -> Table:
        minimal_schema = self._MINIMAL_SCHEMA
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        description_table = get_valid_table(
            table=old_description_table
            if old_description_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
        )

        if self.description_key not in description_table.columns():
            sample = get_dataset_sample_dict(
                to_describe,
                collate_fn=self.collate_fn,
                expected=[self.data_key],
                excluded=[self.description_key],
            )
            sample_data = sample[self.data_key]
            if self.transform is not None:
                sample_data = self.transform(sample_data)

            description = (
                self.extractor.extract_and_fuse(sample_data.to(device=self.device))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            description_table.add_column(
                **{
                    self.description_key: pxt.Array[  # type: ignore[misc]
                        description.shape, pxt.Float
                    ]
                }
            )

        return description_table

    def _distributed_describe(
        self,
        description_table: Table,
        to_describe_dataset: Dataset,
    ) -> Table:
        raise NotImplementedError("Distributed description is not implemented yet.")

    def _sequential_describe(
        self,
        description_table: Table,
        to_describe_dataset: Dataset,
    ) -> Table:
        assert self.batch_size is not None
        assert self.num_workers is not None

        not_empty = (
            description_table.count() > 0
        )  # allow to filter out already described samples

        description_col = get_expr_from_column_name(
            description_table, self.description_key
        )
        already_described = set(
            [
                row["idx"]
                for row in description_table.where(
                    description_col != None  # noqa: E711
                )
                .select(description_table.idx)
                .collect()
            ]
        )

        dataloader = DataLoader(
            to_describe_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        for batch_idx, batch in enumerate(dataloader):
            # Stop if we've processed enough batches
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            # add idx if it is not provided by the dataset
            if "idx" not in batch:
                starts = 0 if not already_described else max(already_described) + 1
                batch["idx"] = [
                    starts + idx for idx in range(len(batch[self.data_key]))
                ]

            # Keep only not yet described samples in the batch
            if not_empty:
                batch = filter_batch_from_indices(
                    batch,
                    already_described,
                )

            if len(batch) == 0:
                continue  # all samples already described

            already_described.update(
                [
                    idx.item() if isinstance(idx, torch.Tensor) else idx
                    for idx in batch["idx"]
                ]
            )

            batch_data = batch[self.data_key]
            if self.transform is not None:
                batch_data = self.transform(batch_data)

            # describe data
            batch[self.description_key] = self.extractor.extract_and_fuse(
                batch_data.to(device=self.device)
            )

            # insert description in the table
            to_insert_keys = [self.description_key]
            if self.to_keep_schema is not None:
                to_insert_keys.extend(self.to_keep_schema.keys())

            include_batch_into_table(
                description_table,
                batch,
                to_insert_keys,
                "idx",
            )

        return description_table
