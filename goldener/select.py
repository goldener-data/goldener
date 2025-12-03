from typing import Callable

import pixeltable as pxt
from pixeltable import Error
from pixeltable.catalog import Table

import torch
import jax.numpy as jnp

from coreax import SquaredExponentialKernel, Data
from coreax.kernels import median_heuristic
from coreax.solvers import KernelHerding
from torch.utils.data import Dataset

from goldener.pxt_utils import (
    set_value_to_idx_rows,
    GoldPxtTorchDataset,
    get_expr_from_column_name,
    get_valid_table,
    include_batch_into_table,
)
from goldener.reduce import GoldReducer
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import filter_batch_from_indices


class GoldSelector:
    """Select a subset of data points from a dataset using coresubset selection.

    The gold runs selection of datapoints from a dataset. The selection is done from a
    coresubset selection algorithm applied on already vectorized representation of the data points.
    If the dataset is too big to fit into memory or the coresubset selection algorithm is too
    computationally expensive, the coresubset selection can be performed in chunks.

    The selection can operate in a sequential (single-process) mode or a
    distributed mode (not implemented). When provided a PyTorch `Dataset`, a
    `collate_fn` can be used to control how samples are batched prior to
    vectorization. The table schema is created/validated automatically and will include
    minimal indexing columns (`idx`, `idx_sample`) required to link vectors back to
    their originating samples.

    During the whole selection process, The computed elements are stored
    in a local PixelTable table column, so that the
    vectorization process is idempotent: calling the same operation multiple times will
    not duplicate or recompute vectors that are already present in the table. This table will at least contain:
        - idx: Index of the vector in the table.
        - sample_idx: Index of the original data point in the dataset.
        - `selection_key`: value assigned to the data point if it has been selected.
        - chunked: Boolean indicating whether the vector has been already processed in a chunk.

    Attributes:
        table_path: Path to store the PixelTable table. It is used to store the vectors
        extracted from the dataset.
        reducer: Optional dimensionality reducer to apply before selection.
        chunk: Optional chunk size for processing data in chunks.
        collate_fn: Optional collate function for the DataLoader.
        vectorized_key: Key pointing to the vector used to make coresubset selection.
        selection_key: Key in which to store the selection value.
        to_keep_schema: Optional schema of additional columns to keep in
        the PixelTable table used to compute the selection.
        batch_size: Batch size for the DataLoader if a dataset is provided for the selection.
        num_workers: Number of workers for the DataLoader if a dataset is provided for the selection.
        allow_existing: Allow to start back the selection from an existing table.
        distribute: Whether to use distributed selection.
        drop_table: Whether to drop the PixelTable table after selection when using `select_in_dataset`.
        max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Int,
        "idx_sample": pxt.Int,
        "chunked": pxt.Bool,
    }

    def __init__(
        self,
        table_path: str,
        reducer: GoldReducer | None = None,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        vectorized_key: str = "vectorized",
        selection_key: str = "selected",
        to_keep_schema: dict[str, type] | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        self.table_path = table_path
        self.reducer = reducer
        self.chunk = chunk
        self.collate_fn = collate_fn
        self.vectorized_key = vectorized_key
        self.selection_key = selection_key
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
        """Select a subset of data points in a dataset.

        The selection is done from a coresubset selection algorithm applied on alreaty vectorized
        representation of the data points. When the chunk attribute is set, the selection is performed in chunks
        to reduce memory consumption. As well, if a reducer is provided,
        the vectors are reduced in dimension before applying the coresubset selection.

        This is a convenience wrapper that runs `select_in_table` to populate
        (or resume populating) the PixelTable table, then wraps the table into a
        `GoldPxtTorchDataset` for downstream consumption. If `drop_table` is True,
        the table will be removed after the dataset is created.

        if `drop_table` is set to True, the PixelTable table used for selection
        will be dropped after the selection dataset is created.

        Args:
            select_from: Dataset or table to select from. If a dataset is provided, each item should be a
            dictionary with at least the `vectorized_key` and `idx_sample` keys after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches.
            If a table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_sample` columns.
            select_count: Number of data points to select. If a table is provided, it should contain at least
            the `vectorized_key` and `idx_sample` columns.
            value: Value to set in the `vectorized_key` column for selected samples.

        Returns:
            A GoldPxtTorchDataset dataset containing the information of the selection table.
            See `select_in_table` for more details.
        """

        selected_table = self.select_in_table(select_from, select_count, value)

        selected_dataset = GoldPxtTorchDataset(selected_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(selected_table)

        return selected_dataset

    def select_in_table(
        self, select_from: Dataset | Table, select_count: int, value: str
    ) -> Table:
        """Select a subset of data points and store their `sample_idx` in a table.

        The selection is done from a coresubset selection algorithm applied on alreaty vectorized
        representation of the data points. When the chunk attribute is set, the selection is performed in chunks
        to reduce memory consumption. As well, if a reducer is provided,
        the vectors are reduced in dimension before applying the coresubset selection.

        This method is idempotent (e.g. failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will restart the selection process
        based on the vectors already present in the PixelTable table.

        Args:
            select_from: Dataset or table to select from. If a dataset is provided, each item should be a
            dictionary with at least the `vectorized_key` and `idx_sample` keys after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches.
            If a table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_sample` columns.
            select_count: Number of data points to select.
            value: Value to set in the `selection_key` column for selected samples.

        Returns:
            A PixelTable Table containing at least the selected sample in the `selection_key` column
            and `idx` and `idx_sample` column as well.
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

        selection_col = get_expr_from_column_name(selection_table, self.selection_key)
        if selection_table.where(selection_col == value).count() == select_count:  # noqa: E712
            return selection_table
        elif self.distribute:
            self._distributed_select(select_from, selection_table, select_count, value)
        else:
            self._sequential_select(select_from, selection_table, select_count, value)

        return selection_table

    def _selection_table_from_table(
        self, select_from: Table, old_selection_table: Table | None
    ) -> Table:
        minimal_schema = self._MINIMAL_SCHEMA

        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

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

        if selection_table.count() != select_from.count():
            self._add_rows_to_selection_table_from_table(select_from, selection_table)

        return selection_table

    def _add_rows_to_selection_table_from_table(
        self,
        select_from: Table,
        selection_table: Table,
    ) -> None:
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
            selection_table.insert([row])

    def _selection_table_from_dataset(
        self, select_from: Dataset, old_selection_table: Table | None
    ) -> Table:
        minimal_schema = self._MINIMAL_SCHEMA
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

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

        self._add_rows_to_selection_table_from_dataset(select_from, selection_table)

        return selection_table

    def _add_rows_to_selection_table_from_dataset(
        self, select_from: Dataset, selection_table: Table
    ) -> None:
        dataloader = torch.utils.data.DataLoader(
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

            include_batch_into_table(
                selection_table,
                batch,
                to_insert_keys,
                "idx",
            )

    def _get_selected_indices(self, table: Table, value: str) -> set[int]:
        return set(
            [
                row["idx_sample"]
                for row in table.where(table.selected == value)  # noqa: E712
                .select(table.idx_sample)
                .distinct()
                .collect()
            ]
        )

    def _sequential_select(
        self,
        select_from: Table,
        selection_table: Table,
        select_count: int,
        value: str,
    ) -> None:
        selection_col = get_expr_from_column_name(selection_table, self.selection_key)
        vectorized_col = get_expr_from_column_name(select_from, self.vectorized_key)

        selection_count = len(self._get_selected_indices(selection_table, value))
        available_for_selection = len(
            [
                row["idx_sample"]
                for row in selection_table.where(selection_col == None)  # noqa: E711
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
            to_chunk_from = selection_table.where(selection_table.selected == None)  # noqa: E711
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
                        & (selection_table.selected == None)  # noqa: E711
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
                    selection_table,
                    selection_table.chunked,
                    set(indices.tolist()),
                    True,
                )

                # make coresubset selection for the chunk
                if self.reducer is not None:
                    vectors = self.reducer.fit_transform(vectors)

                coresubset_indices = self._coresubset_selection(
                    vectors, select_per_chunk, indices
                )

                # update table with selected indices
                set_value_to_idx_rows(
                    selection_table,
                    selection_col,
                    coresubset_indices,
                    value,
                )

                # the sample might have been selected multiple times
                selected_indices = self._get_selected_indices(selection_table, value)
                selection_table.where(
                    selection_table.idx_sample.isin(selected_indices)
                ).update({self.selection_key: value})
                selection_count = len(selected_indices)

    def _distributed_select(
        self,
        select_from: Table,
        selection_table: Table,
        select_count: int,
        value: str,
    ) -> None:
        raise NotImplementedError("Distributed selection is not implemented yet.")

    def _coresubset_selection(
        self, x: torch.Tensor, select_count: int, indices: torch.Tensor
    ) -> set[int]:
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
