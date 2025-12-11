from dataclasses import dataclass
from functools import partial

from pixeltable import Error
from torch.utils.data import RandomSampler, Dataset, DataLoader
from typing_extensions import assert_never
from typing import Callable, Any

from enum import Enum

import torch
from torch import Generator

import pixeltable as pxt
from pixeltable.catalog import Table
import pixeltable.functions as pxtf

from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    get_valid_table,
    get_sample_row_from_idx,
    pxt_torch_dataset_collate_fn,
    get_expr_from_column_name,
    include_batch_into_table,
)
from goldener.torch_utils import make_2d_tensor, get_dataset_sample_dict
from goldener.utils import check_x_and_y_shapes, filter_batch_from_indices


class FilterLocation(Enum):
    """Enum defining filter location strategies for filtering 2D tensor rows.

    START: Filter from the start of the tensor.
    END: Filter from the end of the tensor.
    RANDOM: Filter randomly from the tensor.
    """

    START = "start"
    END = "end"
    RANDOM = "random"


class Filter2DWithCount:
    """Filter 2D tensor rows based on specified criteria.

    Attributes:
        _filter_location: Location to filter from (start, end, random).
        _filter_count: Number of rows to filter.
        _keep: Whether to keep or remove the filtered rows.
        _random_sampler: Sampler for random filtering.
    """

    def __init__(
        self,
        filter_count: int = 1,
        filter_location: FilterLocation = FilterLocation.RANDOM,
        keep: bool = False,
        generator: Generator | None = None,
    ) -> None:
        """Initialize the Filter2DWithCount.

        Args:
            filter_count: Number of rows to filter.
            filter_location: Location to filter from (start, end, random).
            keep: Whether to keep or remove the filtered rows.
            generator: Random number generator for random filtering.
        """
        if filter_count <= 0:
            raise ValueError("filter_count must be greater than 0")

        self._filter_location = filter_location
        self._filter_count = filter_count
        self._keep = keep

        if self._filter_location == FilterLocation.RANDOM:
            self._random_sampler = partial(
                RandomSampler,
                replacement=False,
                num_samples=self._filter_count,
                generator=generator,
            )

    @property
    def is_random(self) -> bool:
        """Check if the filter is random.

        Returns:
            True if the filter uses random sampling, False otherwise.
        """
        return self._filter_location is FilterLocation.RANDOM

    def filter(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Filter a single 2D tensor.

        Args:
            x: Input 2D tensor to filter.

        Returns:
            Filtered tensor.
        """
        return self.filter_tensors({"x": x})["x"]

    def filter_tensors(
        self,
        x: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Filter the input tensor or dictionary of tensors.

        Args:
            x: Input 2D tensor or dictionary of 2D tensors to filter. In case of a dictionary,
            all tensors must have the same number of rows and this number must be greater than filter_count.

        Returns:
            Filtered tensor or dictionary of tensors. If the batch size is less than filter_count,
            the input is returned unchanged.

        Raises:
            ValueError: If input tensors do not have the same batch size or are not 2 dimensional.
        """
        first_tensor = next(iter(x.values()))
        batch_size = first_tensor.shape[0]
        for tensor in x.values():
            if tensor.ndim != 2:
                raise ValueError(
                    "All input tensors must be 2D to filter them with Filter2DWithCount."
                )

            if tensor.shape[0] != batch_size:
                raise ValueError(
                    "All input tensors must have the same batch size to filter them with Filter2DWithCount."
                )

        if batch_size < self._filter_count:
            return x

        if self._filter_location is FilterLocation.START:
            return self._start_filter(x)

        elif self._filter_location is FilterLocation.END:
            return self._end_filter(x)

        elif self._filter_location is FilterLocation.RANDOM:
            return self._random_filter(x)

        else:
            assert_never(self._filter_location)

    def _start_filter(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filter rows from the start of tensors."""
        return (
            {k: v[: self._filter_count] for k, v in x.items()}
            if self._keep
            else {k: v[self._filter_count :] for k, v in x.items()}
        )

    def _end_filter(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filter rows from the end of tensors."""
        return (
            {k: v[-self._filter_count :] for k, v in x.items()}
            if self._keep
            else {k: v[: -self._filter_count] for k, v in x.items()}
        )

    def _random_filter(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filter rows randomly from tensors."""
        first_value = next(iter(x.values()))
        indices = list(self._random_sampler(range(len(first_value))))
        if self._keep:
            return {k: v[indices] for k, v in x.items()}
        else:
            mask = torch.ones(len(first_value), dtype=torch.bool)
            mask[indices] = 0
            return {k: v[mask.bool(), :] for k, v in x.items()}


@dataclass
class Vectorized:
    """Dataclass to hold vectorized tensors and the corresponding batch indices.

    Attributes:
        vectors: 2D tensor of vectorized data.
        batch_indices: 1D tensor containing information about the origin of each vector.
    """

    vectors: torch.Tensor
    batch_indices: torch.Tensor


class TensorVectorizer:
    """Transform input as 2D tensor and filter based on target tensor.

    Attributes:
        keep: Filter2DWithCount instance to keep specific rows in the input `x` of `filter`.
        remove: Filter2DWithCount instance to remove specific rows in the input `x` of `filter`.
        random_filter: Random Filter2DWithCount instance to randomly filter vectors randomly after
        applying `keep`, `remove` and the target `y` on the input `x` of filter.
        transform_y: Optional callable to transform the target tensor before transforming it to 2D.
        channel_pos: position of the channel dimension in the input tensor to vectorize.
    """

    def __init__(
        self,
        keep: Filter2DWithCount | None = None,
        remove: Filter2DWithCount | None = None,
        random_filter: Filter2DWithCount | None = None,
        transform_y: Callable[[torch.Tensor], torch.Tensor] | None = None,
        channel_pos: int = 1,
    ) -> None:
        """Initialize the TensorVectorizer.

        Args:
            keep: Optional filter to keep specific rows in the input.
            remove: Optional filter to remove specific rows from the input.
            random_filter: Optional random filter to apply after keep/remove filters.
            transform_y: Optional transformation to apply to the target tensor.
            channel_pos: Position of the channel dimension in the input tensor. Defaults to 1.

        Raises:
            ValueError: If keep or remove filters are random, or if random_filter is not random.
        """
        self.transform_y = transform_y
        self.channel_pos = channel_pos

        if keep is not None and keep.is_random:
            raise ValueError("The 'keep' filter cannot be random.")
        self.keep = keep

        if remove is not None and remove.is_random:
            raise ValueError("The 'remove' filter cannot be random.")
        self.remove = remove

        if random_filter is not None and not random_filter.is_random:
            raise ValueError("The 'random_filter' must be random.")
        self.random_filter = random_filter

    def vectorize(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> Vectorized:
        """Vectorize input tensor and filter based on target tensor.

        If y is not provided, only filtering and vectorization are performed. If y is provided,
        it is used to filter the vectorized x tensor.

        Args:
            x: Input tensor to vectorize.
            y: Optional target tensor to filter the input tensor.

        Returns:
            Vectorized and filtered input tensor with the information
            about which element of the batch it corresponds to.

        Raises:
            ValueError: If x and y shapes are incompatible. See check_x_and_y_shapes for details.
        """
        if x.ndim < 3:
            raise ValueError("Input tensor x to vectorize must be at least 3D.")

        if y is not None and self.transform_y is not None:
            y = self.transform_y(y)

        if self.channel_pos != 1:
            # Move channel dimension to position 1
            x = x.movedim(self.channel_pos, 1)

        filtered_x = []
        filtered_batch_info = []
        for idx_sample, x_sample in enumerate(x):
            x_sample = x_sample.unsqueeze(0)
            x_sample = make_2d_tensor(x_sample)

            x_sample = self._apply_filter(
                self.keep.filter if self.keep is not None else None, x_sample
            )
            x_sample = self._apply_filter(
                self.remove.filter if self.remove is not None else None, x_sample
            )

            if y is not None:
                y_sample = y[idx_sample].unsqueeze(0)
                x_sample = self._filter_2d_tensors_from_y(x_sample, y_sample)

            filtered_x.append(x_sample)
            filtered_batch_info.append(
                torch.full_like(x_sample[:, 0], idx_sample, dtype=torch.long)
            )

        return Vectorized(
            torch.cat(filtered_x, dim=0), torch.cat(filtered_batch_info, dim=0)
        )

    def _apply_filter(
        self,
        filter: Callable[[torch.Tensor], torch.Tensor] | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a filter function to a tensor if provided.

        Args:
            filter: Optional filter function to apply.
            x: Input tensor.

        Returns:
            Filtered tensor, or original tensor if filter is None.
        """
        if filter is None:
            return x

        return filter(x)

    def _filter_2d_tensors_from_y(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Filter 2D tensor rows based on target tensor.

        When y is not the same batch size as x, it is expanded to match x's batch size.

        Args:
            x: Input 2D tensor to filter. It can be the output of `make_2d_tensor`.
            y: Target tensor to filter the input tensor. It is still in its raw configuration.
            The transform_y callable is applied to y before transforming it to a 2D tensor.

        Returns:
            Filtered input tensor.

        Raises:
            ValueError: If x and y shapes (after 2D transformation) are incompatible.
            See check_x_and_y_shapes for details.
            ValueError: If y tensor after transform contains only zeros.
        """
        y = make_2d_tensor(y)

        y_shape = y.shape
        x_shape = x.shape
        check_x_and_y_shapes(x_shape, y_shape)

        if (y == 0).all():
            raise ValueError(
                "The y tensor after transform must contain at least one "
                "non-zero value to select vectors from x."
            )

        return x[y.bool().squeeze(-1)]


class GoldVectorizer:
    """Extract and flatten vectors from dataset samples and store results in a PixelTable table.

    The GoldVectorizer processes a dataset or PixelTable table to extract and flatten vectors using a
    `TensorVectorizer`. The computed vectors are stored in a local PixelTable table
    (specified by `table_path`) so that the vectorization process is idempotent: calling the
    same operation multiple times will not duplicate or recompute vectors that are already
    present in the table.

    Assuming all the data will not fit in memory, the dataset is processed in batches.
    All torch tensors will be converted to numpy arrays before saving.

    The vectorization can operate in a sequential (single-process) mode or a
    distributed mode (not implemented). The table schema is created/validated automatically and will include
    minimal indexing columns (`idx`, `idx_sample`) required to link vectors back to
    their originating samples.

    Attributes:
        table_path: Path to the PixelTable table where vectorized outputs will be stored locally.
        vectorizer: TensorVectorizer instance for transforming batched inputs into vectors.
        collate_fn: Optional function to collate dataset samples into batches composed of
            dictionaries with at least the key specified by `data_key` returning a PyTorch Tensor.
            If None, the dataset is expected to directly provide such batches.
        data_key: Key in the batch dictionary that contains the data to vectorize. Default is "features".
        target_key: Optional key in the batch dictionary containing the target used to filter vectors. Default is "target".
        vectorized_key: Column name to store the resulting vectors in the PixelTable table. Default is "vectorized".
        to_keep_schema: Optional dictionary defining additional columns to keep from the original dataset/table
            into the vectorized table. The keys are the column names and the values are the PixelTable types.
        batch_size: Batch size used when iterating over the data. Defaults to 1 if not distributed.
        num_workers: Number of workers for the PyTorch DataLoader during iteration on data. Defaults to 0 if not distributed.
        allow_existing: If False, an error will be raised when the table already exists. Default is True.
        distribute: Whether to use distributed processing for vectorization and table population. Not implemented yet. Default is False.
        drop_table: Whether to drop the vectorized table after creating the dataset with vectorized outputs. It is only applied
            when using `vectorize_in_dataset`. Default is False.
        max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {"idx": pxt.Int, "idx_sample": pxt.Int}

    def __init__(
        self,
        table_path: str,
        vectorizer: TensorVectorizer,
        collate_fn: Callable | None = None,
        data_key: str = "features",
        target_key: str = "target",
        vectorized_key: str = "vectorized",
        to_keep_schema: dict[str, type] | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        """Initialize the GoldVectorizer.

        Args:
            table_path: Path to the PixelTable table for storing vectors.
            vectorizer: TensorVectorizer instance for transforming tensors.
            collate_fn: Optional collate function for preparing batches.
            data_key: Key for data in the batch dictionary. Defaults to "features".
            target_key: Key for target in the batch dictionary. Defaults to "target".
            vectorized_key: Column name for storing vectors. Defaults to "vectorized".
            to_keep_schema: Optional schema for additional columns to preserve.
            batch_size: Batch size for processing. Defaults to 1 if not distributed.
            num_workers: Number of workers. Defaults to 0 if not distributed.
            allow_existing: Whether to allow using an existing table. Defaults to True.
            distribute: Whether to use distributed processing. Defaults to False.
            drop_table: Whether to drop the table after dataset creation. Defaults to False.
            max_batches: Optional maximum number of batches to process.
        """
        self.table_path = table_path
        self.vectorizer = vectorizer
        self.collate_fn = collate_fn
        self.data_key = data_key
        self.target_key = target_key
        self.vectorized_key = vectorized_key
        self.to_keep_schema = to_keep_schema
        self.allow_existing = allow_existing
        self.distribute = distribute
        self.drop_table = drop_table
        self.max_batches = max_batches

        self.batch_size: int | None
        self.num_workers: int | None
        if not self.distribute:
            self.batch_size = 1 if batch_size is None else batch_size
            self.num_workers = 0 if num_workers is None else num_workers
        else:
            self.batch_size = batch_size
            self.num_workers = num_workers

    def vectorize_in_dataset(
        self,
        to_vectorize: Dataset | Table,
    ) -> GoldPxtTorchDataset:
        """Extract and flatten vectors from samples and return results as a GoldPxtTorchDataset.

        The vectorization process extracts and flattens vectors from the provided dataset or table
        using the `TensorVectorizer` instance and stores them in a PixelTable table specified by `table_path`.
        The resulting vectors are stored in the column specified by `vectorized_key`.

        This is a convenience wrapper that runs `vectorize_in_table` to populate
        (or resume populating) the PixelTable table, then wraps the table into a
        `GoldPxtTorchDataset` for downstream consumption. If `drop_table` is True,
        the table will be removed after the dataset is created.

        Args:
            to_vectorize: Dataset or Table to be vectorized. If a Dataset is provided, each item should be a
                dictionary with at least the key specified by `data_key` after applying the collate_fn.
                If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
                it should contain both 'idx' and `data_key` columns.

        Returns:
            A GoldPxtTorchDataset containing at least the vectorized data in the `vectorized_key` key
                and `idx` and `idx_sample` keys as well.
        """
        vectorized_table = self.vectorize_in_table(to_vectorize)

        vectorized_dataset = GoldPxtTorchDataset(vectorized_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(vectorized_table)

        return vectorized_dataset

    def vectorize_in_table(
        self,
        to_vectorize: Dataset | Table,
    ) -> Table:
        """Extract and flatten vectors from samples and store results in a PixelTable table.

        The vectorization process extracts and flattens vectors from the provided dataset or table
        using the `TensorVectorizer` instance and stores them in a PixelTable table specified by `table_path`.
        The resulting vectors are stored in the column specified by `vectorized_key`.

        This method is idempotent (i.e., failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will not duplicate or recompute the vectorization
        already present in the PixelTable table.

        Args:
            to_vectorize: Dataset or Table to be vectorized. If a Dataset is provided, each item should be a
                dictionary with at least the key specified by `data_key` after applying the collate_fn.
                If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
                it should contain both 'idx' and `data_key` columns.

        Returns:
            A PixelTable Table containing at least the vectorized data in the `vectorized_key` column
                and `idx` and `idx_sample` columns as well.
        """

        # If the computation was already started or already done, we resume from there
        try:
            old_vectorized_table = pxt.get_table(
                self.table_path,
                if_not_exists="ignore",
            )
        except Error:
            old_vectorized_table = None

        if not self.allow_existing and old_vectorized_table is not None:
            raise ValueError(
                f"Table at path {self.table_path} already exists and "
                "allow_existing is set to False."
            )

        # get the table and dataset to execute the vectorize pipeline
        to_vectorize_dataset: Dataset | GoldPxtTorchDataset
        if isinstance(to_vectorize, Table):
            vectorized_table = self._vectorized_table_from_table(
                to_vectorize=to_vectorize,
                old_vectorized_table=old_vectorized_table,
            )

            if vectorized_table.count() > 0 and "idx" in to_vectorize.columns():
                to_vectorize_indices = set(
                    [
                        row["idx"]
                        for row in to_vectorize.select(to_vectorize.idx).collect()
                    ]
                )
                already_vectorized = set(
                    [
                        row["idx_sample"]
                        for row in vectorized_table.select(vectorized_table.idx_sample)
                        .distinct()
                        .collect()
                    ]
                )
                if not to_vectorize_indices.difference(already_vectorized):
                    return vectorized_table

            to_vectorize_dataset = GoldPxtTorchDataset(to_vectorize)
        else:
            to_vectorize_dataset = to_vectorize
            vectorized_table = self._vectorized_table_from_dataset(
                to_vectorize, old_vectorized_table
            )

        if self.distribute:
            vectorized = self._distributed_vectorize(
                vectorized_table, to_vectorize_dataset
            )
        else:
            vectorized = self._sequential_vectorize(
                vectorized_table, to_vectorize_dataset
            )

        return vectorized

    def _vectorized_table_from_table(
        self, to_vectorize: Table, old_vectorized_table: Table | None
    ) -> Table:
        """Create or validate the vectorized table schema from a PixelTable table.

        This private method sets up the table structure and adds the vectorized column
        with the appropriate array type based on the vectorizer's output shape.

        Args:
            to_vectorize: The source PixelTable table to vectorize.
            old_vectorized_table: Existing vectorized table if resuming, or None.

        Returns:
            The vectorized table with proper schema.
        """
        minimal_schema = self._MINIMAL_SCHEMA
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        vectorized_table = get_valid_table(
            table=old_vectorized_table
            if old_vectorized_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
        )

        if self.vectorized_key not in vectorized_table.columns():
            sample = get_sample_row_from_idx(
                to_vectorize,
                collate_fn=pxt_torch_dataset_collate_fn,
                expected_keys=[self.data_key],
            )
            vectorized = self.vectorizer.vectorize(
                sample[self.data_key],
                sample.get(self.target_key, None),
            ).vectors[0]
            vectorized_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized.shape, pxt.Float
                    ]
                }
            )

        return vectorized_table

    def _vectorized_table_from_dataset(
        self, to_vectorize: Dataset, old_vectorized_table: Table | None
    ) -> Table:
        """Create or validate the vectorized table schema from a PyTorch Dataset.

        This private method sets up the table structure and adds the vectorized column
        with the appropriate array type based on the vectorizer's output shape.

        Args:
            to_vectorize: The source PyTorch Dataset to vectorize.
            old_vectorized_table: Existing vectorized table if resuming, or None.

        Returns:
            The vectorized table with proper schema.
        """
        minimal_schema = self._MINIMAL_SCHEMA
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        vectorized_table = get_valid_table(
            table=old_vectorized_table
            if old_vectorized_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
        )

        if self.vectorized_key not in vectorized_table.columns():
            sample = get_dataset_sample_dict(
                to_vectorize,
                collate_fn=self.collate_fn,
                expected=[self.data_key],
                excluded=[self.vectorized_key],
            )

            target = sample.get(self.target_key, None)
            if target is not None and self.collate_fn is not None:
                assert isinstance(target, torch.Tensor)
                target = target.unsqueeze(0)

            vectorized = (
                self.vectorizer.vectorize(
                    sample[self.data_key].unsqueeze(0)
                    if self.collate_fn is None
                    else sample[self.data_key],
                    target,
                )
                .vectors[0]
                .detach()
                .cpu()
                .numpy()
            )
            vectorized_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized.shape, pxt.Float
                    ]
                }
            )

        return vectorized_table

    def _distributed_vectorize(
        self,
        vectorized_table: Table,
        to_vectorize_dataset: Dataset,
    ) -> Table:
        """Run distributed vectorization process (not implemented).

        Args:
            vectorized_table: The table to store vectorized outputs.
            to_vectorize_dataset: The dataset to vectorize.

        Returns:
            The populated vectorized table.

        Raises:
            NotImplementedError: Always raised as distributed mode is not yet implemented.
        """
        raise NotImplementedError("Distributed Vectorization is not implemented yet.")

    def _sequential_vectorize(
        self,
        vectorized_table: Table,
        to_vectorize_dataset: Dataset,
    ) -> Table:
        """Run sequential (single-process) vectorization process.

        This private method processes the dataset in batches, applies vectorization using
        the TensorVectorizer, and stores the results in the vectorized table. It is idempotent
        and will skip samples that have already been vectorized.

        Args:
            vectorized_table: The table to store vectorized outputs.
            to_vectorize_dataset: The dataset to vectorize.

        Returns:
            The populated vectorized table.
        """
        assert self.batch_size is not None
        assert self.num_workers is not None

        not_empty = (
            vectorized_table.count() > 0
        )  # allow to filter out already described samples

        vectorized_col = get_expr_from_column_name(
            vectorized_table, self.vectorized_key
        )
        already_vectorized = set(
            [
                row["idx_sample"]
                for row in vectorized_table.where(
                    vectorized_col != None  # noqa: E711
                )
                .select(vectorized_table.idx_sample)
                .collect()
            ]
        )

        dataloader = DataLoader(
            to_vectorize_dataset,
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
                starts = 0 if not already_vectorized else max(already_vectorized) + 1
                batch["idx"] = [
                    starts + idx for idx in range(len(batch[self.data_key]))
                ]

            # Keep only not yet described samples in the batch
            if not_empty:
                batch = filter_batch_from_indices(
                    batch,
                    already_vectorized,
                )

                if len(batch) == 0:
                    continue  # all samples already described

            already_vectorized.update(
                [
                    idx.item() if isinstance(idx, torch.Tensor) else idx
                    for idx in batch["idx"]
                ]
            )

            # describe data
            vectorized = self.vectorizer.vectorize(
                batch[self.data_key],
                batch.get(self.target_key, None),
            )

            max_idx = [
                row["max"]
                for row in vectorized_table.select(
                    pxtf.max(vectorized_table.idx)  # type: ignore[call-arg]
                ).collect()
            ][0]

            to_keep_keys = None
            if self.to_keep_schema is not None:
                to_keep_keys = list(self.to_keep_schema.keys())

            batch = self._unwrap_vectors_in_batch(
                vectorized=vectorized,
                batch=batch,
                starts=max_idx + 1 if max_idx is not None else 0,
                to_keep_keys=to_keep_keys,
            )

            # insert vectorized in the table
            to_insert_keys = [self.vectorized_key, "idx_sample"]
            if to_keep_keys is not None:
                to_insert_keys.extend(to_keep_keys)

            include_batch_into_table(
                vectorized_table,
                batch,
                to_insert_keys,
                "idx",
            )

        return vectorized_table

    def _unwrap_vectors_in_batch(
        self,
        vectorized: Vectorized,
        batch: dict[str, Any],
        starts: int = 0,
        to_keep_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Unwrap vectorized output into individual rows for table insertion.

        This private method converts the Vectorized dataclass (which contains batched vectors
        and their batch indices) into a dictionary format suitable for inserting into the
        PixelTable table, with one entry per vector.

        Args:
            vectorized: Vectorized output containing vectors and batch indices.
            batch: Original batch dictionary with metadata.
            starts: Starting index for assigning new idx values.
            to_keep_keys: Optional list of additional keys to preserve from the batch.

        Returns:
            Dictionary with unwrapped vectors and metadata, ready for table insertion.
        """
        new_batch: dict[str, Any] = {
            "idx": [],
            self.vectorized_key: [],
            "idx_sample": [],
        }
        if to_keep_keys is not None:
            for key in to_keep_keys:
                new_batch[key] = []

        sample_indices = [
            (idx_value.item() if isinstance(idx_value, torch.Tensor) else idx_value)
            for batch_idx, idx_value in enumerate(batch["idx"])
        ]

        vectorized_idx = vectorized.batch_indices
        for vec_idx, vector in enumerate(vectorized.vectors):
            new_batch[self.vectorized_key].append(vector)
            new_batch["idx"].append(starts + vec_idx)
            batch_idx = vectorized_idx[vec_idx].item()
            assert isinstance(batch_idx, int)
            new_batch["idx_sample"].append(sample_indices[batch_idx])
            if to_keep_keys is not None:
                for key in to_keep_keys:
                    new_batch[key].append(batch[key][batch_idx])

        return new_batch
