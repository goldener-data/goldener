from typing import Callable, Literal, Any

import torch

import pixeltable as pxt
from pixeltable.catalog import Table
from torch.utils.data import Dataset

from goldener.extract import GoldFeatureExtractor
from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    get_expr_from_column_name,
    get_sample_row_from_idx,
    pxt_torch_dataset_collate_fn,
    get_table_if_column_started,
    get_valid_view_for_table,
    get_table_from_dataset,
)


class GoldDescriptor:
    """Describe the `data` of a dataset by extracting features from a pretrained model.

    Assuming all the data will not fit in memory, the dataset is processed in batches.

    All the data of the dataset, the computed features included, will be saved in a local Pixeltable table during
    the computation. In this table, the `data` of the dataset will be saved in the shape and scale obtained
    after applying the `collate_fn` if provided. These arrays are expected to be all the same size.
    All torch tensors will be converted to numpy arrays before saving.

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
        batch_size: Optional batch size for processing the dataset.
        num_workers: Optional number of worker threads for data loading.
        if_exists: Behavior if the table already exists ('error' or 'replace_force'). If 'replace_force',
        the existing table will be replaced, otherwise an error will be raised.
        distribute: Whether to use distributed processing for feature extraction and table population. Not implemented yet.
        drop_table: Whether to drop the description table after creating the dataset with descriptions. It is only applied
        when using `describe_in_dataset`.
        device: Torch device to use for feature extraction. If None, it will use 'cuda' if available, otherwise 'cpu'.
        max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
    """

    def __init__(
        self,
        table_path: str,
        extractor: GoldFeatureExtractor,
        transform: Callable | None = None,
        collate_fn: Callable | None = None,
        data_key: str = "data",
        description_key: str = "features",
        batch_size: int | None = None,
        num_workers: int | None = None,
        if_exists: Literal["error", "replace_force"] = "error",
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
        self.if_exists = if_exists
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

        This method is idempotent (e.g. failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will not duplicate or recompute the descriptions
        already present in the PixelTable table.

        Args:
            to_describe: Dataset or Table to be described. If a Dataset is provided, each item should be a
            dictionary with at least the key specified by `data_key` after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
            it should contain both 'idx' and `data_key` columns.

        Returns:
            A GoldPxtTorchDataset containing the original data and the extracted features.
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

        This method is idempotent (e.g. failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will not duplicate or recompute the descriptions
        already present in the PixelTable table.

        Args:
            to_describe: Dataset or Table to be described. If a Dataset is provided, each item should be a
            dictionary with at least the key specified by `data_key` after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
            it should contain both 'idx' and `data_key` column. The table containing
            the description of the data will be created as a view of this table.

        Returns:
            A PixelTable Table containing the original data and the extracted features.
        """
        # If the computation was already started or already done, we resume from there
        old_description_table = get_table_if_column_started(
            self.table_path, self.description_key, True
        )

        # get the table and dataset to execute the description pipeline
        to_describe_dataset: GoldPxtTorchDataset | Dataset
        if isinstance(to_describe, Table):
            description_table = self._description_table_from_table(
                to_describe, old_description_table
            )
            description_col = get_expr_from_column_name(
                description_table, self.description_key
            )
            with_no_description = description_table.where(description_col == None)  # noqa: E711
            if with_no_description.count() == 0:
                return description_table

            table_indices = [
                row["idx"]
                for row in with_no_description.select(description_table.idx).collect()
            ]

            to_describe_dataset = GoldPxtTorchDataset(
                to_describe.where(description_table.idx.isin(table_indices))
            )
        else:
            to_describe_dataset = to_describe
            description_table = self._description_table_from_dataset(
                to_describe_dataset, old_description_table
            )

        # run the description process
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
        description_table = get_valid_view_for_table(
            table=to_describe,
            view=old_description_table
            if old_description_table is not None
            else self.table_path,
            expected=[self.data_key, "idx"],
            excluded=[self.description_key],
        )

        if self.description_key not in description_table.columns():
            sample = get_sample_row_from_idx(
                description_table,
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
        self, dataset: Dataset, old_description_table: Table | None
    ) -> Table:
        if old_description_table is None:
            description_table = get_table_from_dataset(
                table_path=self.table_path,
                dataset=dataset,
                collate_fn=self.collate_fn,
                expected=[self.data_key],
                excluded=[self.description_key],
                if_exists=self.if_exists,
            )
        else:
            # resume from existing table
            # An error will be raised by Pixeltable insert if the data is not in accordance with
            # the existing columns
            description_table = old_description_table

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

        only_description = description_table.get_metadata()[
            "is_view"
        ]  # if the table is a view, only the description column can be updated
        not_empty = (
            description_table.count() > 0
        )  # allow to filter out already described samples

        description_col = get_expr_from_column_name(
            description_table, self.description_key
        )
        already_described = [
            row["idx"]
            for row in description_table.where(
                description_col != None  # noqa: E711
            )
            .select(description_table.idx)
            .collect()
        ]

        dataloader = torch.utils.data.DataLoader(
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
                starts = self.batch_size * batch_idx
                batch["idx"] = [
                    starts + idx for idx in range(len(batch[self.data_key]))
                ]

            # Keep only not yet described samples in the batch
            if not_empty:
                batch = self._batch_for_description(
                    batch,
                    already_described,
                )

            if len(batch) == 0:
                continue  # all samples already described

            already_described.extend(
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
            self._insert_batch_description(
                description_table,
                batch,
                only_description,
            )

        return description_table

    def _batch_for_description(
        self,
        batch: dict[str, Any],
        already_described: list[int],
    ) -> dict[str, Any]:
        keep_in_batch = [
            idx_position
            for idx_position, idx_value in enumerate(batch["idx"])
            if (idx_value.item() if isinstance(idx_value, torch.Tensor) else idx_value)
            not in already_described
        ]
        if not keep_in_batch:
            return {}  # all samples already described

        def filter_batched_values(
            batched_value: list | torch.Tensor,
        ) -> list | torch.Tensor:
            """Inner function to remove already described samples from the batch."""
            filtered = [
                value
                for idx_value, value in enumerate(batched_value)
                if idx_value in keep_in_batch
            ]
            if isinstance(batched_value, torch.Tensor):
                return torch.stack(filtered, dim=0)
            else:
                return filtered

        return {
            key: filter_batched_values(batched_value)
            for key, batched_value in batch.items()
        }

    def _insert_batch_description(
        self,
        description_table: Table,
        batch: dict[str, Any],
        only_description: bool,
    ) -> None:
        if only_description:
            for batch_idx, idx_value in enumerate(batch["idx"]):
                description_table.where(
                    description_table.idx == idx_value.item()
                    if isinstance(idx_value, torch.Tensor)
                    else idx_value
                ).update(
                    {
                        self.description_key: (
                            batch[self.description_key][batch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    }
                )
        else:
            if (
                description_table.where(
                    description_table.idx.isin(batch["idx"])
                ).count()
                > 0
            ):
                raise ValueError(
                    "Description table already contains some of the indices in the current batch. "
                    "This should not happen when describing from a dataset."
                )

            to_insert = [
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
                for sample_idx in range(len(batch["idx"]))
            ]
            description_table.insert(to_insert)
