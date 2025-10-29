from typing import Callable, Literal

import pixeltable as pxt
from pixeltable.catalog import Table

import torch
import jax.numpy as jnp

from coreax import SquaredExponentialKernel, Data
from coreax.kernels import median_heuristic
from coreax.solvers import KernelHerding
from torch.utils.data import Dataset

from goldener.pxt_utils import create_pxt_table_from_sample, set_value_to_idx_rows
from goldener.reduce import GoldReducer
from goldener.torch_utils import ResetableTorchIterableDataset
from goldener.vectorize import GoldVectorizer


class GoldSelector:
    """Selector class to select a subset of data points from a dataset using coresubset selection.

    If the dataset is too big to fit into memory or the coresubset selection algorithm is too
    computationally expensive, the coresubset selection can be performed in chunks.
    During the whole selection process, the vectors extracted from the dataset are stored
    in a PixelTable table. This table will contain the following columns:
        - idx: Index of the vector in the table.
        - vector: The vector representation of the data point.
        - sample_idx: Index of the original data point in the dataset.
        - selected: Boolean indicating whether the vector has been selected.
        - chunked: Boolean indicating whether the vector has been processed in a chunk.

    Attributes:
        table_path: Path to store the PixelTable table. It is used to store the vectors
        extracted from the dataset.
        vectorizer: Vectorizer to convert data points into vectors.
        select_key: Key in the dataset sample dictionary to select data points from.
        select_target_key: Optional key in the dataset sample dictionary to select target values from.
        It allows to filter out some vectors based on their target values.
        drop_table: Whether to drop the PixelTable table after selection.
        reducer: Optional dimensionality reducer to apply before selection.
        chunk: Optional chunk size for processing data in chunks.
        collate_fn: Optional collate function for the DataLoader.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of workers for the DataLoader.
        if_exists: Behavior if the PixelTable table already exists. Options are 'error' or 'replace_force'.
        distribute: Whether to use distributed selection.
        shuffle: Whether to shuffle the dataset during loading.
        generator: Optional random number generator for shuffling.
    """

    def __init__(
        self,
        table_path: str,
        vectorizer: GoldVectorizer,
        select_key: str = "features",
        select_target_key: str | None = None,
        drop_table: bool = False,
        reducer: GoldReducer | None = None,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        if_exists: Literal["error", "replace_force"] = "error",
        distribute: bool = False,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.table_path = table_path
        self.vectorizer = vectorizer
        self.select_key = select_key
        self.select_target_key = select_target_key
        self.drop_table = drop_table
        self.reducer = reducer
        self.chunk = chunk
        self.collate_fn = collate_fn
        self.if_exists = if_exists
        self.distribute = distribute
        self.shuffle = shuffle
        self.generator = generator

        self.batch_size: int | None
        self.num_workers: int | None

        if not self.distribute:
            self.batch_size = 1 if batch_size is None else batch_size
            self.num_workers = 0 if num_workers is None else num_workers
        else:
            self.batch_size = batch_size
            self.num_workers = num_workers

    def select(self, dataset: Dataset, select_count: int) -> set[int]:
        """Select a subset of data points from the dataset.

        The selection is done from a coresubset selection algorithm applied on the vectorized
        representation of the data points (filtered with the target if provided). When the chunk
        attribute is set, the selection is performed in chunks to reduce memory consumption. As well,
        if a reducer is provided, the vectors are reduced in dimension before applying the coresubset selection.

        All along the selection process, the vectors are stored in a PixelTable table located at table_path.
        If drop_table is set to True, the table is deleted after the selection.

        Args:
            dataset: Dataset to select from. Each item should be a dictionary with
            at least the `select_key` and `idx` keys after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches.
            select_count: Number of data points to select.
        """
        self._check_dataset(dataset)

        if self.distribute:
            selected = self._distributed_select(dataset, select_count)
        else:
            selected = self._sequential_select(dataset, select_count)

        if self.drop_table:
            pxt.drop_table(self.table_path)

        return selected

    def _get_selected_indices(self, table: Table) -> set[int]:
        return set(
            [
                row["sample_idx"]
                for row in table.where(table.selected == True)  # noqa: E712
                .select(table.sample_idx)
                .distinct()
                .collect()
            ]
        )

    def _sequential_select(self, dataset: Dataset, select_count: int) -> set[int]:
        vector_table = self._sequential_store_vectors_in_table(dataset)

        # The coresubset selection is done from all the patches (after filtering) of all data point.
        # Then, the same data point can be selected multiple times if it has multiple patches selected.
        # To achieve select_count of unique data points, we loop until we have enough unique data points selected.
        already_selected = 0
        while already_selected < select_count:
            # select only rows still not selected
            to_chunk_from = vector_table.where(vector_table.selected == False)  # noqa: E712
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
            select_per_chunk = (select_count - already_selected) // chunk_loop_count
            if select_per_chunk == 0:
                select_per_chunk = 1

            # make coresubset selection per chunk
            for chunk_idx in range(chunk_loop_count):
                if already_selected >= select_count:
                    break

                # select data for the current chunk among vector not yet selected
                to_select_from = vector_table.where(
                    (vector_table.chunked == False) & (vector_table.selected == False)  # noqa: E712
                ).select(vector_table.vector, vector_table.idx)
                if chunk_idx < chunk_loop_count - 1:
                    to_select_from = to_select_from.sample(chunk_size)

                # load the vectors and the corresponding indices for the chunk
                to_select = [
                    (
                        torch.from_numpy(sample["vector"]),
                        torch.tensor(sample["idx"]).unsqueeze(0),
                    )
                    for sample in to_select_from.collect()
                ]
                vectors_list, indices_list = map(list, zip(*to_select))
                vectors = torch.stack(vectors_list, dim=0)
                indices = torch.cat(indices_list, dim=0)
                set_value_to_idx_rows(
                    vector_table,
                    vector_table.chunked,
                    set(indices.tolist()),
                    True,
                )  # selected indices are marked as already chunked

                # make coresubset selection for the chunk
                if self.reducer is not None:
                    vectors = self.reducer.fit_transform(vectors)

                coresubset_indices = self._coresubset_selection(
                    vectors, select_per_chunk, indices
                )

                # update table with selected indices
                set_value_to_idx_rows(
                    vector_table,
                    vector_table.selected,
                    coresubset_indices,
                    True,
                )

                already_selected = len(self._get_selected_indices(vector_table))

        return self._get_selected_indices(vector_table)

    def _distributed_select(self, dataset: Dataset, select_count: int):
        raise NotImplementedError("Distributed selection is not implemented yet.")

    def _check_dataset(self, dataset: Dataset) -> None:
        sample = (
            next(dataset)
            if isinstance(dataset, ResetableTorchIterableDataset)
            else dataset[0]
        )
        if isinstance(dataset, ResetableTorchIterableDataset):
            dataset.reset()

        if self.collate_fn is not None:
            sample = self.collate_fn([sample])

        if not isinstance(sample, dict):
            raise ValueError(
                "Dataset sample must be a dictionary after applying the collate_fn."
            )

        if self.select_key not in sample:
            raise ValueError(f"Dataset sample must contain a {self.select_key} key.")
        elif not isinstance(sample[self.select_key], torch.Tensor):
            raise ValueError(
                f"Value of {self.select_key} must correspond to a torch.Tensor."
            )

        if "idx" not in sample:
            raise ValueError("Dataset must have an 'idx' key.")

        if self.select_target_key is not None:
            if self.select_target_key not in sample:
                raise ValueError(
                    f"Dataset sample must contain a {self.select_target_key} key."
                )
            elif not isinstance(sample[self.select_target_key], torch.Tensor):
                raise ValueError(
                    f"Value of {self.select_target_key} must correspond to a torch.Tensor."
                )

    def _initialize_table(
        self, dataset: Dataset | ResetableTorchIterableDataset
    ) -> Table:
        sample = (
            next(dataset)
            if isinstance(dataset, ResetableTorchIterableDataset)
            else dataset[0]
        )
        if isinstance(dataset, ResetableTorchIterableDataset):
            dataset.reset()

        if self.collate_fn is not None:
            sample = self.collate_fn([sample])
        elif sample[self.select_key].ndim == 2:
            sample[self.select_key] = sample[self.select_key].unsqueeze(0)

        vectorized = self.vectorizer.vectorize(
            sample[self.select_key],
            None
            if self.select_target_key is None
            else sample[self.select_target_key].unsqueeze(0),
        )

        unique_row = {
            "vector": vectorized.vectors[0].numpy(),
            "sample_idx": vectorized.batch_indices[0].item(),
            "idx": 0,
            "selected": False,
            "chunked": False,
        }

        pxt_table = create_pxt_table_from_sample(
            self.table_path, unique_row, self.if_exists
        )
        pxt_table.where(pxt_table.idx == 0).delete()  # remove the initial sample

        return pxt_table

    def _sequential_store_vectors_in_table(self, dataset: Dataset) -> Table:
        pxt_table = self._initialize_table(dataset)

        assert self.batch_size is not None
        assert self.num_workers is not None
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            generator=self.generator,
        )

        vector_count = 0
        for batch_idx, batch in enumerate(data_loader):
            vectors = batch[self.select_key]

            vectorized = self.vectorizer.vectorize(vectors)
            to_insert = [
                {
                    "idx": vector_idx + vector_count,
                    "vector": vector.numpy(),
                    "sample_idx": batch["idx"][sample_idx].item(),
                    "selected": False,
                    "chunked": False,
                }
                for vector_idx, (sample_idx, vector) in enumerate(
                    zip(vectorized.batch_indices, vectorized.vectors)
                )
            ]
            pxt_table.insert(to_insert)
            vector_count += len(to_insert)

        return pxt_table

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
