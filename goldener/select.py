from typing import Callable

import pixeltable as pxt
from pixeltable import Error
from pixeltable.catalog import Table

import torch
import jax.numpy as jnp

from coreax import SquaredExponentialKernel, Data
from coreax.kernels import median_heuristic
from coreax.solvers import KernelHerding
from torch.utils.data import Dataset, DataLoader

from goldener.pxt_utils import (
    set_value_to_idx_rows,
    GoldPxtTorchDataset,
    get_expr_from_column_name,
    get_valid_table,
    include_batch_into_table,
    get_column_distinct_ratios,
)
from goldener.reduce import GoldReducer
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import filter_batch_from_indices


class GoldSelector:
    """Select a subset of data points from vectorized samples and store results in a PixelTable table.

    The GoldSelector processes a dataset or PixelTable table to perform coresubset selection using a
    kernel herding algorithm on already vectorized representations. The selection results are stored
    in a local PixelTable table (specified by `table_path`) so that the selection process is idempotent:
    calling the same operation multiple times will not duplicate or recompute selections that are already
    present in the table.

    If the dataset is too big to fit into memory or the coresubset selection algorithm is too
    computationally expensive, the coresubset selection can be performed in chunks.
    All torch tensors will be converted to numpy arrays before saving.

    The selection can operate in a sequential (single-process) mode or a
    distributed mode (not implemented). The table schema is created/validated automatically and will include
    minimal indexing columns (`idx`, `idx_sample`) required to link selected samples back to
    their originating data points.

    Attributes:
        table_path: Path to the PixelTable table where selection results will be stored locally.
        reducer: Optional GoldReducer instance for dimensionality reduction before selection.
        chunk: Optional chunk size for processing data in chunks to reduce memory consumption.
        collate_fn: Optional function to collate dataset samples into batches composed of
            dictionaries with at least the key specified by `vectorized_key` returning a PyTorch Tensor.
            If None, the dataset is expected to directly provide such batches.
        vectorized_key: Key in the batch dictionary that contains the vectorized data for selection. Default is "vectorized".
        selection_key: Column name to store the selection value in the PixelTable table. Default is "selected".
        class_key: Optional key for class-based stratified selection.
        to_keep_schema: Optional dictionary defining additional columns to keep from the original dataset/table
            into the selection table. The keys are the column names and the values are the PixelTable types.
        batch_size: Batch size used when iterating over the data. Defaults to 1 if not distributed.
        num_workers: Number of workers for the PyTorch DataLoader during iteration on data. Defaults to 0 if not distributed.
        allow_existing: If False, an error will be raised when the table already exists. Default is True.
        distribute: Whether to use distributed processing for selection and table population. Not implemented yet. Default is False.
        drop_table: Whether to drop the selection table after creating the dataset with selection results. It is only applied
            when using `select_in_dataset`. Default is False.
        max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Int,
        "idx_sample": pxt.Int,
    }

    def __init__(
        self,
        table_path: str,
        reducer: GoldReducer | None = None,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        vectorized_key: str = "vectorized",
        selection_key: str = "selected",
        class_key: str | None = None,
        to_keep_schema: dict[str, type] | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        """Initialize the GoldSelector.

        Args:
            table_path: Path to store the PixelTable table.
            reducer: Optional dimensionality reducer to apply before selection.
            chunk: Optional chunk size for processing data in chunks.
            collate_fn: Optional collate function for the DataLoader.
            vectorized_key: Key pointing to the vector for selection. Defaults to "vectorized".
            selection_key: Key for storing selection values. Defaults to "selected".
            class_key: Optional key for class stratification.
            to_keep_schema: Optional schema for additional columns to preserve.
            batch_size: Batch size for processing. Defaults to 1 if not distributed.
            num_workers: Number of workers. Defaults to 0 if not distributed.
            allow_existing: Whether to allow using an existing table. Defaults to True.
            distribute: Whether to use distributed selection. Defaults to False.
            drop_table: Whether to drop the table after dataset creation. Defaults to False.
            max_batches: Optional maximum number of batches to process.
        """
        self.table_path = table_path
        self.reducer = reducer
        self.chunk = chunk
        self.collate_fn = collate_fn
        self.vectorized_key = vectorized_key
        self.selection_key = selection_key
        self.class_key = class_key
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

    def select_in_dataset(
        self, select_from: Dataset | Table, select_count: int, value: str
    ) -> GoldPxtTorchDataset:
        """Select a subset of samples using coresubset selection and return results as a GoldPxtTorchDataset.

        The selection process applies a coresubset selection algorithm on already vectorized
        representations of the data points and stores results in a PixelTable table specified by `table_path`.
        When the chunk attribute is set, the selection is performed in chunks to reduce memory consumption.
        If a reducer is provided, the vectors are reduced in dimension before applying the coresubset selection.

        This is a convenience wrapper that runs `select_in_table` to populate
        (or resume populating) the PixelTable table, then wraps the table into a
        `GoldPxtTorchDataset` for downstream consumption. If `drop_table` is True,
        the table will be removed after the dataset is created.

        Args:
            select_from: Dataset or Table to select from. If a Dataset is provided, each item should be a
                dictionary with at least the `vectorized_key` and `idx_sample` keys after applying the collate_fn.
                If the collate_fn is None, the dataset is expected to directly provide such batches.
                If a Table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_sample` columns.
            select_count: Number of data points to select.
            value: Value to set in the `selection_key` column for selected samples.

        Returns:
            A GoldPxtTorchDataset containing at least the selection information in the `selection_key` key
                and `idx` and `idx_sample` keys as well.
        """

        selected_table = self.select_in_table(select_from, select_count, value)

        selected_dataset = GoldPxtTorchDataset(selected_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(selected_table)

        return selected_dataset

    def select_in_table(
        self, select_from: Dataset | Table, select_count: int, value: str | None
    ) -> Table:
        """Select a subset of samples using coresubset selection and store results in a PixelTable table.

        The selection process applies a coresubset selection algorithm on already vectorized
        representations of the data points and stores results in a PixelTable table specified by `table_path`.
        When the chunk attribute is set, the selection is performed in chunks to reduce memory consumption.
        If a reducer is provided, the vectors are reduced in dimension before applying the coresubset selection.

        This method is idempotent (i.e., failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will restart the selection process
        based on the vectors already present in the PixelTable table.

        Args:
            select_from: Dataset or Table to select from. If a Dataset is provided, each item should be a
                dictionary with at least the `vectorized_key` and `idx_sample` keys after applying the collate_fn.
                If the collate_fn is None, the dataset is expected to directly provide such batches.
                If a Table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_sample` columns.
            select_count: Number of data points to select.
            value: Value to set in the `selection_key` column for selected samples.

        Returns:
            A PixelTable Table containing at least the selection information in the `selection_key` column
                and `idx` and `idx_sample` columns as well.
        """
        try:
            old_selection_table = pxt.get_table(
                self.table_path,
                if_not_exists="ignore",
            )
        except Error:
            old_selection_table = None

        if not self.allow_existing and old_selection_table is not None:
            raise ValueError(
                f"Table at path {self.table_path} already exists and "
                "allow_existing is set to False."
            )

        if isinstance(select_from, Table):
            selection_table = self._selection_table_from_table(
                select_from=select_from,
                old_selection_table=old_selection_table,
            )
        else:
            selection_table = self._selection_table_from_dataset(
                select_from, old_selection_table
            )
            select_from = selection_table

        assert isinstance(select_from, Table)

        if (
            len(
                self.get_selected_sample_indices(
                    selection_table, value, self.selection_key
                )
            )
            == select_count
        ):
            return selection_table
        elif self.distribute:
            self._distributed_select(select_from, selection_table, select_count, value)
        else:
            self._sequential_select(select_from, selection_table, select_count, value)

        return selection_table

    def _selection_table_from_table(
        self, select_from: Table, old_selection_table: Table | None
    ) -> Table:
        """Create or validate the selection table schema from a PixelTable table.

        This private method sets up the table structure with necessary columns for tracking
        selection status and ensures all rows from the source table are represented.

        Args:
            select_from: The source PixelTable table to select from.
            old_selection_table: Existing selection table if resuming, or None.

        Returns:
            The selection table with proper schema and initial rows.
        """
        minimal_schema = self._MINIMAL_SCHEMA

        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.class_key is not None:
            minimal_schema[self.class_key] = pxt.String

        selection_table = get_valid_table(
            table=old_selection_table
            if old_selection_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
        )

        if self.vectorized_key not in select_from.columns():
            raise ValueError(
                f"Table at path {self.table_path} does not contain "
                f"the required column {self.vectorized_key}."
            )

        if self.selection_key not in selection_table.columns():
            selection_table.add_column(
                if_exists="error", **{self.selection_key: pxt.String}
            )

        if "chunked" not in selection_table.columns():
            selection_table.add_column(if_exists="error", **{"chunked": pxt.Bool})
            selection_table.update({"chunked": False})

        if selection_table.count() > 0:
            to_select_indices = set(
                [
                    row["idx"]
                    for row in select_from.select(select_from.idx).distinct().collect()
                ]
            )
            already_in_selection = set(
                [
                    row["idx"]
                    for row in selection_table.select(selection_table.idx)
                    .distinct()
                    .collect()
                ]
            )
            still_to_select = to_select_indices.difference(already_in_selection)
            if not still_to_select:
                return selection_table

        self._add_rows_to_selection_table_from_table(select_from, selection_table)

        return selection_table

    def _add_rows_to_selection_table_from_table(
        self,
        select_from: Table,
        selection_table: Table,
    ) -> None:
        """Add rows from the source table to the selection table.

        This private method populates the selection table with rows from the source table,
        preserving necessary columns and initializing selection status.

        Args:
            select_from: The source PixelTable table.
            selection_table: The selection table to populate.
        """
        col_list = [
            "idx_sample",
            "idx",
        ]
        if self.to_keep_schema is not None:
            col_list.extend(list(self.to_keep_schema.keys()))

        if self.selection_key in select_from.columns():
            col_list.append(self.selection_key)

        for idx_row, row in enumerate(
            select_from.select(
                *[get_expr_from_column_name(select_from, col) for col in col_list]
            ).collect()
        ):
            if self.max_batches is not None:
                if self.batch_size is None:
                    raise ValueError("batch_size must be set when max_batches is used.")
                if idx_row >= self.batch_size * self.max_batches:
                    break

            if selection_table.where(selection_table.idx == row["idx"]).count() > 0:
                continue  # already included

            if self.selection_key not in row:
                row[self.selection_key] = None

            if "chunked" not in row:
                row["chunked"] = False

            selection_table.insert([row])

    def _selection_table_from_dataset(
        self, select_from: Dataset, old_selection_table: Table | None
    ) -> Table:
        """Create or validate the selection table schema from a PyTorch Dataset.

        This private method sets up the table structure with necessary columns including
        the vectorized column with proper array type based on the dataset sample.

        Args:
            select_from: The source PyTorch Dataset to select from.
            old_selection_table: Existing selection table if resuming, or None.

        Returns:
            The selection table with proper schema.
        """
        minimal_schema = self._MINIMAL_SCHEMA
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.class_key is not None:
            minimal_schema[self.class_key] = pxt.String

        selection_table = get_valid_table(
            table=old_selection_table
            if old_selection_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
        )

        if self.vectorized_key not in selection_table.columns():
            sample = get_dataset_sample_dict(
                select_from,
                collate_fn=self.collate_fn,
                expected=[self.vectorized_key],
            )

            vectorized_value = sample[self.vectorized_key].detach().cpu().numpy()
            selection_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized_value.shape, pxt.Float
                    ]
                }
            )

        if self.selection_key not in selection_table.columns():
            selection_table.add_column(
                if_exists="error", **{self.selection_key: pxt.String}
            )

        if "chunked" not in selection_table.columns():
            selection_table.add_column(if_exists="error", **{"chunked": pxt.Bool})
            selection_table.update({"chunked": False})

        self._add_rows_to_selection_table_from_dataset(select_from, selection_table)

        return selection_table

    def _add_rows_to_selection_table_from_dataset(
        self, select_from: Dataset, selection_table: Table
    ) -> None:
        """Add rows from the source dataset to the selection table.

        This private method iterates through the dataset in batches and populates the
        selection table with vectorized data and metadata, skipping already processed samples.

        Args:
            select_from: The source PyTorch Dataset.
            selection_table: The selection table to populate.
        """
        dataloader = DataLoader(
            select_from,
            batch_size=self.batch_size if self.batch_size is not None else 1,
            num_workers=self.num_workers if self.num_workers is not None else 1,
            collate_fn=self.collate_fn,
        )

        vectorized_col = get_expr_from_column_name(selection_table, self.vectorized_key)
        already_included = set(
            [
                row["idx"]
                for row in selection_table.where(
                    vectorized_col != None  # noqa: E711
                )
                .select(selection_table.idx)
                .collect()
            ]
        )
        not_empty = (
            selection_table.count() > 0
        )  # allow to filter out already described samples

        for batch_idx, batch in enumerate(dataloader):
            # Stop if we've processed enough batches
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            if "idx" not in batch:
                assert self.batch_size is not None
                starts = batch_idx * self.batch_size
                batch["idx"] = [
                    starts + idx for idx in range(len(batch[self.vectorized_key]))
                ]

            if "chunked" not in batch:
                batch["chunked"] = [
                    False for _ in range(len(batch[self.vectorized_key]))
                ]

            # Keep only not yet included samples in the batch
            if not_empty:
                batch = filter_batch_from_indices(
                    batch,
                    already_included,
                )

                if len(batch) == 0:
                    continue  # all samples already described

            if self.selection_key not in batch:
                batch[self.selection_key] = [
                    None for _ in range(len(batch[self.vectorized_key]))
                ]

            already_included.update(
                [
                    idx.item() if isinstance(idx, torch.Tensor) else idx
                    for idx in batch["idx"]
                ]
            )
            to_insert_keys = [self.vectorized_key, "idx_sample", self.selection_key]
            if self.to_keep_schema is not None:
                to_insert_keys.extend(list(self.to_keep_schema.keys()))
            if self.class_key is not None:
                to_insert_keys.append(self.class_key)

            include_batch_into_table(
                selection_table,
                batch,
                to_insert_keys,
                "idx",
            )

    @staticmethod
    def get_selected_sample_indices(
        table: Table,
        value: str | None,
        selection_key: str,
        class_key: str | None = None,
        class_value: str | None = None,
        idx_key: str = "idx_sample",
    ) -> set[int]:
        """Get the indices of samples selected with a given value.

        Args:
            table: PixelTable table to query.
            value: Value in the selection_key column to filter selected samples.
            selection_key: Column name used to store selection values.
            class_key: Optional column name used to filter samples by class.
            class_value: Optional class value to filter samples by class.
            idx_key: Column name used to get sample indices.
        """
        idx_col = get_expr_from_column_name(table, idx_key)
        selection_col = get_expr_from_column_name(table, selection_key)
        if class_value is not None and class_key is not None:
            class_col = get_expr_from_column_name(table, class_key)
            query = (selection_col == value) & (class_col == class_value)  # noqa: E712
        else:
            if class_key is not None or class_value is not None:
                raise ValueError("class_key and class_value must be set together.")
            query = selection_col == value  # noqa: E712

        return set(
            [
                row[idx_key]
                for row in table.where(query)  # noqa: E712
                .select(idx_col)
                .distinct()
                .collect()
            ]
        )

    def _sequential_select(
        self,
        select_from: Table,
        selection_table: Table,
        select_count: int,
        value: str | None,
    ) -> None:
        """Run sequential (single-process) selection process.

        This private method handles class-stratified selection if a class_key is configured,
        otherwise performs selection on the full dataset. It delegates the actual coresubset
        selection to _class_select.

        Args:
            select_from: The source table with vectorized data.
            selection_table: The table to store selection results.
            select_count: Number of samples to select.
            value: Value to assign to selected samples in the selection_key column.
        """
        if self.class_key is not None:
            class_col = get_expr_from_column_name(selection_table, self.class_key)
            class_ratios = get_column_distinct_ratios(selection_table, class_col)

            for class_idx, (class_value, class_ratio) in enumerate(
                class_ratios.items()
            ):
                already_selected = len(
                    self.get_selected_sample_indices(
                        table=selection_table,
                        value=value,
                        selection_key=self.selection_key,
                        class_key=self.class_key,
                        class_value=class_value,
                    )
                )
                if class_idx < len(class_ratios) - 1:
                    class_count = int(select_count * class_ratio)
                else:
                    # The last class takes the missing samples count to avoid rounding issues
                    other_classes = len(
                        self.get_selected_sample_indices(
                            table=selection_table,
                            value=value,
                            selection_key=self.selection_key,
                        )
                    )
                    class_count = select_count - other_classes

                if class_count == 0:
                    raise ValueError(
                        f"Class '{class_value}' has ratio {class_ratio} which results in zero samples "
                        f"for the requested select_count of {select_count}. "
                    )

                class_count = class_count - already_selected
                if class_count == 0:
                    continue
                elif class_count < 0:
                    raise ValueError(
                        "The size of the selection table has decreased since the 1st selection computation"
                    )

                self._class_select(
                    select_from,
                    selection_table,
                    class_count,
                    value,
                    class_value=class_value,
                )

        else:
            self._class_select(
                select_from,
                selection_table,
                select_count,
                value,
            )

    def _class_select(
        self,
        select_from: Table,
        selection_table: Table,
        select_count: int,
        value: str | None,
        class_value: str | None = None,
    ) -> None:
        """Perform coresubset selection for a specific class or all data.

        This private method implements chunked coresubset selection using kernel herding.
        It processes data in chunks to manage memory, applies optional dimensionality reduction,
        and updates the selection table with selected samples.

        Args:
            select_from: The source table with vectorized data.
            selection_table: The table to store selection results.
            select_count: Number of samples to select.
            value: Value to assign to selected samples in the selection_key column.
            class_value: Optional class value to filter samples by class.
        """
        selection_col = get_expr_from_column_name(selection_table, self.selection_key)
        vectorized_col = get_expr_from_column_name(select_from, self.vectorized_key)

        selection_count = len(
            self.get_selected_sample_indices(
                table=selection_table,
                value=value,
                selection_key=self.selection_key,
                class_key=self.class_key,
                class_value=class_value,
            )
        )

        if class_value is not None:
            assert self.class_key is not None
            class_col = get_expr_from_column_name(selection_table, self.class_key)
            available_query = (selection_col == None) & (class_col == class_value)  # noqa: E712 E711
        else:
            available_query = selection_col == None  # noqa: E711

        available_for_selection = len(
            [
                row["idx_sample"]
                for row in selection_table.where(available_query)  # noqa: E711
                .select(selection_table.idx_sample)
                .distinct()
                .collect()
            ]
        )

        if available_for_selection < (select_count - selection_count):
            raise ValueError(
                "Cannot select more unique data points than available in the dataset."
            )

        # The coresubset selection is done from all the vectors (after filtering) of all data point
        # (depending on data, a data point can have multiple vectors).
        # Then, the same data point can be selected multiple times if it has multiple vectors selected.
        # To achieve select_count of unique data points, we loop until we have enough unique data points selected.
        while selection_count < select_count:
            # select only rows still not selected
            to_chunk_from = selection_table.where(available_query)
            to_chunk_from.update(
                {"chunked": False}
            )  # unchunk all rows not yet selected

            # initialize the chunk settings: chunk size, number of chunks, selection per chunk
            to_chunk_from_count = to_chunk_from.count()
            chunk_size = (
                to_chunk_from_count
                if self.chunk is None
                else min(self.chunk, to_chunk_from_count)
            )
            chunk_loop_count = to_chunk_from_count // chunk_size
            select_per_chunk = (select_count - selection_count) // chunk_loop_count
            if select_per_chunk == 0:
                select_per_chunk = 1

            # make coresubset selection per chunk
            for chunk_idx in range(chunk_loop_count):
                if selection_count >= select_count:
                    break

                # select data for the current chunk among vector not yet selected
                not_chunked_indices = [
                    row["idx"]
                    for row in selection_table.where(
                        (selection_table.chunked == False)  # noqa: E712
                        & available_query
                    )
                    .select(selection_table.idx)
                    .collect()
                ]

                to_select_from = select_from.where(
                    select_from.idx.isin(not_chunked_indices)
                ).select(vectorized_col, select_from.idx)
                if chunk_idx < chunk_loop_count - 1:
                    to_select_from = to_select_from.sample(chunk_size)

                # load the vectors and the corresponding indices for the chunk
                to_select = [
                    (
                        torch.from_numpy(sample[self.vectorized_key]),
                        torch.tensor(sample["idx"]).unsqueeze(0),
                    )
                    for sample in to_select_from.collect()
                ]
                vectors_list, indices_list = map(list, zip(*to_select))
                vectors = torch.stack(vectors_list, dim=0)
                indices = torch.cat(indices_list, dim=0)

                # selected indices are marked as already chunked
                set_value_to_idx_rows(
                    table=selection_table,
                    col_expr=selection_table.chunked,
                    idx_expr=selection_table.idx_sample,
                    indices=set(indices.tolist()),
                    value=True,
                )

                # make coresubset selection for the chunk
                if self.reducer is not None:
                    vectors = self.reducer.fit_transform(vectors)

                coresubset_indices = self._coresubset_selection(
                    vectors, select_per_chunk, indices
                )

                # update table with selected indices
                set_value_to_idx_rows(
                    table=selection_table,
                    col_expr=selection_col,
                    idx_expr=selection_table.idx,
                    indices=coresubset_indices,
                    value=value,
                )

                # the sample might have been selected multiple times
                selected_indices = self.get_selected_sample_indices(
                    table=selection_table,
                    value=value,
                    selection_key=self.selection_key,
                    class_key=self.class_key,
                    class_value=class_value,
                )
                selection_table.where(
                    selection_table.idx_sample.isin(selected_indices)
                ).update({self.selection_key: value})
                selection_count = len(selected_indices)

    def _distributed_select(
        self,
        select_from: Table,
        selection_table: Table,
        select_count: int,
        value: str | None,
    ) -> None:
        """Run distributed selection process (not implemented).

        Args:
            select_from: The source table with vectorized data.
            selection_table: The table to store selection results.
            select_count: Number of samples to select.
            value: Value to assign to selected samples in the selection_key column.

        Raises:
            NotImplementedError: Always raised as distributed mode is not yet implemented.
        """
        raise NotImplementedError("Distributed selection is not implemented yet.")

    def _coresubset_selection(
        self, x: torch.Tensor, select_count: int, indices: torch.Tensor
    ) -> set[int]:
        """Apply kernel herding coresubset selection algorithm.

        This private method uses the coreax library's KernelHerding solver with a
        SquaredExponentialKernel to select a diverse subset of vectors.

        Args:
            x: Input vectors to select from.
            select_count: Number of vectors to select.
            indices: Original indices corresponding to each vector.

        Returns:
            Set of selected indices from the original index space.
        """
        herding_solver = KernelHerding(
            select_count,
            kernel=SquaredExponentialKernel(
                length_scale=float(median_heuristic(jnp.asarray(x.mean(1).numpy())))
            ),
        )
        herding_coreset, _ = herding_solver.reduce(Data(jnp.array(x.numpy())))  # type: ignore[arg-type]

        return set(
            indices[torch.tensor(herding_coreset.unweighted_indices.tolist())].tolist()
        )
