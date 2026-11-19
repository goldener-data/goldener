"""Compute density-based vector scores and retrieve per-sample weights.

Use this stage to give larger weights to vectors in sparse neighborhoods.
Tensor tools compute raw scores; dataset and table orchestration persist
those scores for later retrieval.

- `GoldDensityWeighingTool` defines the tensor interface for scoring a full
  population or query batches against that population.
- `GoldKnnDensityWeighingTool` implements the interface using mean nearest-neighbor
  distance, an absolute distance floor, and a configurable exponent. It can
  optionally ignore other vectors from the same sample.
- `GoldDensityWeigher` orchestrates scoring of Dataset or PixelTable inputs and
  persists raw scores per vector. Its `get_weights_per_idx` method retrieves
  vector scores or aggregates them into sample weights, optionally capping raw
  scores at a maximum and normalizing them.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from logging import getLogger
from statistics import mean
from typing import Any, Callable, Literal

import numpy as np
import pixeltable as pxt
import torch
from pixeltable import Error, Query
from pixeltable.catalog import Table
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from goldener.clusterize import get_random_chunk_assignment
from goldener.do import GoldDoerWithTable
from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    check_pxt_table_has_primary_key,
    get_expr_from_column_name,
    get_valid_table,
    make_batch_ready_for_table,
    get_sample_row_from_idx,
)
from goldener.reduce import GoldReductionTool, GoldReductionToolWithFit
from goldener.distance import DistanceType
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import (
    validate_positive_number,
    validate_positive_integer,
    filter_batch_from_indices,
    validate_vectorized_input,
)

logger = getLogger(__name__)


class GoldDensityWeighingTool(ABC):
    """Compute one nonnegative density score for each vector in a 2D tensor.

    The input matrix has shape `(num_vectors, feature_dim)` and implementations
    return a one-dimensional tensor with one score per queried row.
    `weigh_batch` scores query rows against the full input population.

    Attributes:
        device: Device used for tensor computation. Defaults to `"cpu"`.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        """Initialize the density weighing tool.

        Args:
            device: Torch device used for computation. Defaults to CPU.
        """
        self.device = torch.device(device)

    @abstractmethod
    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        """Return one density score per row of the reference population.

        Args:
            x: Reference tensor with shape `(num_vectors, feature_dim)`.

        Returns:
            A tensor of finite nonnegative scores with shape `(num_vectors,)`, in
            input row order.
        """

    @abstractmethod
    def weigh_batch(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Score batch rows against the full reference population.

        Args:
            x: Reference tensor shaped `(num_vectors, feature_dim)`.
            batch: Rows from `x` shaped `(batch_size, feature_dim)`, preserving
                query order and duplicates. One copy of each query is excluded
                from its neighbors.

        Returns:
            One finite nonnegative score per batch row, in query order.
        """


class GoldKnnDensityWeighingTool(GoldDensityWeighingTool):
    """Score vectors by their mean distance to the nearest neighbors.

    Each score is `max(mean_neighbor_distance, eps) ** power`. Distances before
    exponentiation use the selected metric's original units; larger scores
    indicate sparser neighborhoods when `power` is positive. `eps` floors the
    mean distance, including for identical vectors. By default the search keeps
    one extra neighbor and drops it, so the reference population needs at least
    `k + 2` vectors. With `exclude_same_sample`, every reference vector that
    shares the query sample `idx` is ineligible, and each query needs `k`
    references from other samples. Identical vectors from different samples stay
    eligible. The tool is deterministic and has no random state.

    Attributes:
        k: Positive number of neighbors retained per row. Defaults to `5`.
            The default search retrieves `k + 1` entries and discards one nearest
            entry, which requires at least `k + 2` vectors. Same-sample exclusion
            instead keeps the `k` nearest references from other samples.
        power: Finite nonnegative exponent applied to the mean neighbor
            distance. Defaults to `0.5`.
        distance_func: Pairwise distance callable selected by `distance`.
            Defaults to the Euclidean distance function.
        distance_batch_size: Positive number of query rows compared per batch
            against the full input population. Defaults to `1024`.
        eps: Finite positive floor on the mean neighbor distance before
            exponentiation. Defaults to `1e-12`.
        device: Torch device used for computation. Defaults to CPU; returned
            scores retain the query tensor's device and dtype.
        exclude_same_sample: Whether to ignore every reference vector whose
            sample `idx` equals the query's. Defaults to `False`, which excludes
            only one copy of the query vector and ignores supplied sample ids.
    """

    def __init__(
        self,
        k: int = 5,
        power: float = 0.5,
        distance: DistanceType = DistanceType.EUCLIDEAN,
        distance_batch_size: int = 1024,
        eps: float = 1e-12,
        device: torch.device | str = "cpu",
        exclude_same_sample: bool = False,
    ) -> None:
        """Configure deterministic k-nearest-neighbor density scores.

        Args:
            k: Positive integer number of neighbors per query. The default search
                retrieves `k + 1` entries and discards one nearest entry to
                exclude the query itself. Reference populations then require at
                least `k + 2` vectors. With `exclude_same_sample`, the `k`
                neighbors are chosen after removing every reference with the
                query sample `idx`. Defaults to `5`.
            power: Finite nonnegative exponent applied after flooring the mean
                neighbor distance. Defaults to `0.5`.
            distance: Distance metric used to compare vectors. Defaults to Euclidean.
            distance_batch_size: Positive integer number of query rows compared per
                distance batch. Every batch uses the full reference population.
                Defaults to `1024`.
            eps: Finite positive floor on the mean neighbor distance before
                exponentiation. Defaults to `1e-12`.
            device: Computation device. Defaults to CPU.
            exclude_same_sample: Whether to exclude every reference vector with
                the same sample `idx` as the query. Defaults to `False`.

        Raises:
            ValueError: If `k` or `distance_batch_size` is not a positive integer,
                `power` is not finite and nonnegative, `eps` is not finite and
                positive, or `exclude_same_sample` is not a boolean. Booleans are
                not accepted for the numeric parameters.
        """
        super().__init__(device)
        validate_positive_integer(k, "k")
        validate_positive_integer(distance_batch_size, "distance_batch_size")
        validate_positive_number(power, "power", allow_zero=True)
        validate_positive_number(eps, "eps")
        if not isinstance(exclude_same_sample, bool):
            raise ValueError("exclude_same_sample must be a boolean.")
        self.k = k
        self.power = float(power)
        self.distance_func = distance.distance_function
        self.distance_batch_size = distance_batch_size
        self.eps = float(eps)
        self.exclude_same_sample = exclude_same_sample

    def weigh(
        self, x: torch.Tensor, sample_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return powered kNN distance scores for every input vector.

        Each score is `max(mean_neighbor_distance, eps) ** power`. By default one
        nearest entry is excluded per query. With `exclude_same_sample`, every
        reference row whose sample `idx` matches that query is excluded instead.
        Distances are computed in `torch.float64` on the configured device.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)`, with
                positive feature width. The default mode requires at least
                `k + 2` rows.
            sample_idx: Sample id for each row of `x`, shaped `(num_vectors,)`.
                Required when `exclude_same_sample` is true, because these rows
                are both the reference and the queries. Ignored otherwise.

        Returns:
            Scores shaped `(num_vectors,)`, in input row order and with the input
            tensor's dtype and device.

        Raises:
            ValueError: If `x` is not a finite floating 2D tensor with positive
                feature width, the default mode has fewer than `k + 2` rows, or
                same-sample exclusion is missing identities or eligible neighbors.
        """
        return self.weigh_batch(
            x, x, sample_idx=sample_idx, query_sample_idx=sample_idx
        )

    def weigh_batch(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        sample_idx: torch.Tensor | None = None,
        query_sample_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score query rows using the reference rows as potential neighbors.

        Each score is `max(mean_neighbor_distance, eps) ** power`. By default one
        nearest entry is discarded per query to exclude itself, retaining other
        identical neighbors. With `exclude_same_sample`, masking uses the
        supplied sample ids: a reference is ineligible when its id equals the
        query id, and the `k` nearest eligible references are kept. Identical
        vectors from different samples remain eligible. Distances are computed
        in `torch.float64` on the configured device.

        Args:
            x: Finite floating reference tensor shaped `(num_vectors, feature_dim)`,
                with positive feature width. The default mode requires at least
                `k + 2` rows.
            batch: Finite floating query tensor shaped `(batch_size, feature_dim)`.
                Queries are rows scored against `x`; membership is not checked.
                Order and duplicates are preserved, and feature width must match `x`.
            sample_idx: Sample id for each reference row, shaped `(num_vectors,)`.
                Required when `exclude_same_sample` is true. Ignored otherwise.
            query_sample_idx: Sample id for each query row, shaped `(batch_size,)`.
                Required when `exclude_same_sample` is true, including repeated
                query rows. Ignored otherwise.

        Returns:
            Scores shaped `(batch_size,)`, with the query tensor's dtype and device.
            An empty query batch returns an empty tensor after the population and
            identity checks above.

        Raises:
            ValueError: If either input is not a finite floating 2D tensor with
                positive feature width, feature widths differ, the default mode
                has fewer than `k + 2` reference rows, or same-sample exclusion
                is missing identities or a query has fewer than `k` eligible
                references.
        """
        validate_vectorized_input(x, "GoldDensityWeighingTool")
        if not self.exclude_same_sample and len(x) < self.k + 2:
            raise ValueError(
                f"x must contain at least k + 2 ({self.k + 2}) vectors; got {len(x)}."
            )
        validate_vectorized_input(batch, "GoldDensityWeighingTool batch")

        if batch.shape[1] != x.shape[1]:
            raise ValueError("batch must have the same feature width as x.")

        if self.exclude_same_sample:
            sample_idx = self._validate_sample_idx(sample_idx, len(x), "sample_idx")
            query_sample_idx = self._validate_sample_idx(
                query_sample_idx, len(batch), "query_sample_idx"
            )

        if len(batch) == 0:
            return batch.new_empty((0,))

        reference = x.to(device=self.device, dtype=torch.float64)
        queries = batch.to(device=self.device, dtype=torch.float64)
        means = []
        query_parts = queries.split(self.distance_batch_size)
        if self.exclude_same_sample:
            assert query_sample_idx is not None and sample_idx is not None
            id_parts = query_sample_idx.split(self.distance_batch_size)
        else:
            id_parts = tuple(None for _ in query_parts)

        offset = 0
        for query, query_ids in zip(query_parts, id_parts, strict=True):
            distances = self.distance_func(query, reference)
            if self.exclude_same_sample:
                assert query_ids is not None and sample_idx is not None
                means.append(
                    self._mean_eligible_distances(
                        distances, sample_idx, query_ids, offset
                    )
                )
            else:
                neighbors = distances.topk(self.k + 1, largest=False, dim=1).values
                means.append(neighbors[:, 1:].mean(dim=1))
            offset += len(query)

        return torch.cat(means).clamp_min(self.eps).pow(self.power).to(batch)

    @staticmethod
    def _validate_sample_idx(
        sample_idx: torch.Tensor | None, n_rows: int, name: str
    ) -> torch.Tensor:
        """Return one integer sample id per vector row.

        Args:
            sample_idx: Candidate sample identities.
            n_rows: Number of vector rows those identities must match.
            name: Argument name included in the error message.

        Returns:
            The validated identity tensor.

        Raises:
            ValueError: If the identities are missing or are not a 1D integer
                tensor of length `n_rows`.
        """
        if sample_idx is None:
            raise ValueError(f"{name} is required when exclude_same_sample is True.")
        if (
            not isinstance(sample_idx, torch.Tensor)
            or sample_idx.ndim != 1
            or sample_idx.dtype
            not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            )
        ):
            shape = getattr(sample_idx, "shape", None)
            dtype = getattr(sample_idx, "dtype", type(sample_idx).__name__)
            raise ValueError(
                f"{name} must be a 1D integer tensor with one sample idx per row; "
                f"got shape {shape} and dtype {dtype}."
            )
        if sample_idx.shape[0] != n_rows:
            raise ValueError(
                f"{name} must contain one sample idx per row; "
                f"got {sample_idx.shape[0]} entries for {n_rows} rows."
            )
        return sample_idx

    def _mean_eligible_distances(
        self,
        distances: torch.Tensor,
        sample_idx: torch.Tensor,
        query_sample_idx: torch.Tensor,
        query_offset: int,
    ) -> torch.Tensor:
        """Average the `k` nearest references from other samples.

        Args:
            distances: Pairwise distances shaped `(num_queries, num_references)`.
            sample_idx: Sample id of each reference column.
            query_sample_idx: Sample id of each query row in this distance batch.
            query_offset: Position of this batch's first query in the full query
                tensor. Error messages use that position.

        Returns:
            Mean eligible distance for each query row, shaped `(num_queries,)`.

        Raises:
            ValueError: If a query has fewer than `k` references with a different
                sample id.
        """
        reference_ids = sample_idx.to(device=distances.device)
        query_ids = query_sample_idx.to(device=distances.device)
        eligible = query_ids[:, None] != reference_ids[None, :]
        eligible_count = eligible.sum(dim=1)
        short = eligible_count < self.k
        if bool(short.any()):
            local = int(torch.nonzero(short, as_tuple=False)[0])
            raise ValueError(
                "Each query needs at least k "
                f"({self.k}) eligible reference vectors after same-sample "
                f"exclusion; query row {query_offset + local} has "
                f"{int(eligible_count[local])}."
            )
        masked = distances.masked_fill(~eligible, torch.inf)
        return masked.topk(self.k, largest=False, dim=1).values.mean(dim=1)


class GoldDensityWeigher(GoldDoerWithTable):
    """Weigh vectorized samples using their neighborhood density.

    The GoldDensityWeigher processes a dataset or PixelTable table to compute
    one nonnegative score per vector using a density weighing tool. Results are
    stored in a local PixelTable table (specified by `table_path`) so that the
    weighing process is idempotent: repeated calls reuse completed scores
    without duplicating or recomputing them. Inputs and scoring settings must
    remain unchanged when resuming an existing result table.

    If the data is too large to fit into memory or the weighing algorithm is
    too computationally expensive, weighing can be performed in chunks.

    Weighing operates in sequential (single-process) mode; distributed
    processing is not implemented and requesting it raises NotImplementedError.
    The table schema is created or validated automatically and includes `idx`
    and `idx_vector` to link scores back to their source samples and vectors.
    When `GoldKnnDensityWeighingTool.exclude_same_sample` is true, neighbor
    search ignores every reference vector that shares the query sample `idx`.
    The weigher reads the `idx` stored for each `idx_vector` and passes that
    pair through reduction, chunking, and query batching in the vector row
    order. A reducer must keep that order; the weigher checks the row count.
    The default tool still excludes only one copy of the query vector and
    requires `k + 2` vectors in each scored population.

    Attributes:
        table_path: Path to the PixelTable table where weighing results will be stored locally.
        weigher: Tensor scoring tool. Defaults to `GoldKnnDensityWeighingTool()`.
            Supplied non-null source scores are retained without recomputation.
        weighing_batch_size: Positive integer query batch size, or `None` (default)
            to call `weigh` once per chunk. Every query batch uses the full chunk.
        reducer: Optional dimensionality reduction tool. Defaults to `None`.
            A fittable reducer is applied with `fit_transform` on each chunk;
            a transform-only reducer is applied with `transform`.
        chunk: Positive integer maximum number of pending vectors per independent
            chunk. `None` (default) uses one pending population per label.
        collate_fn: Optional function to collate dataset samples into batches composed of
            dictionaries with at least the key specified by `vectorized_key` returning a PyTorch Tensor.
            If None, `collate_keeping_sequences_as_sequences` is used, which preserves sequence-valued
            fields as one sequence per sample.
        vectorized_key: Key in the batch dictionary that contains the vectorized data for weighing. Default is "vectorized".
        include_vectorized_in_table: Whether to include the vectorized data in the weighing table. Defaults to False.
            It is only applied if the weighing table is created from a Table (it is forced anyway for Dataset).
        weight_key: Column name to store the weight value in the PixelTable table. Default is "weight".
        label_key: Optional key for label stratification. Default is None.
        to_keep_schema: Optional dictionary defining additional columns to keep from the original dataset/table
            into the weighing table. The keys are the column names and the values are the PixelTable types.
        min_pxt_insert_size: Minimum number of rows to accumulate before inserting into PixelTable. Default is 100.
        batch_size: Batch size used for dataset ingestion.
        num_workers: Number of workers for the PyTorch DataLoader during iteration on data.
        allow_existing: If False, an error will be raised when the table already exists. Default is True.
        distribute: Whether to use distributed processing for weighing and table population. Not implemented yet. Default is False.
        drop_table: Whether to drop the weighing table after creating the dataset with weighing results. It is only applied
            when using `weigh_in_dataset`. Default is False.
        max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
        random_state: Optional random state for reproducibility during chunk assignment. Default is None.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
        "idx_vector": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        table_path: str,
        weigher: GoldDensityWeighingTool = GoldKnnDensityWeighingTool(),
        weighing_batch_size: int | None = None,
        reducer: GoldReductionTool | None = None,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        vectorized_key: str = "vectorized",
        weight_key: str = "weight",
        to_keep_schema: dict[str, type] | None = None,
        min_pxt_insert_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 0,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
        random_state: int | None = None,
        include_vectorized_in_table: bool = False,
        label_key: str | None = None,
    ) -> None:
        """Initialize the GoldDensityWeigher.

        Args:
            table_path: Path to store the PixelTable table.
            weigher: Tensor scoring tool. Defaults to `GoldKnnDensityWeighingTool()`.
            weighing_batch_size: Positive integer query batch size, or `None` (default)
                to score each chunk in one call. Each query batch uses the full chunk.
            reducer: Optional GoldReductionTool instance for dimensionality reduction before weighing. A
                fittable reducer is applied with `fit_transform` on each chunk;
                a transform-only reducer is applied with `transform`.
            chunk: Optional maximum number of vectors per independently scored chunk.
                If None, all vectors ingested by one call are placed in one chunk.
            collate_fn: Optional collate function for the DataLoader.
                If None, `collate_keeping_sequences_as_sequences` is used.
            vectorized_key: Key pointing to the vector for weighing. Defaults to "vectorized".
            include_vectorized_in_table: Whether to include the vectorized data in the weighing table. Defaults to False.
                It is only applied if the weighing table is created from a Table (it is forced anyway for Dataset).
            weight_key: Key for storing raw per-vector scores. Defaults to "weight".
            label_key: Optional key for label stratification. Default is None.
            to_keep_schema: Optional schema for additional columns to preserve.
            min_pxt_insert_size: Minimum number of rows to accumulate before inserting into PixelTable. Default is 100.
            batch_size: Batch size used for dataset ingestion. Defaults to 1.
            num_workers: Number of workers for the PyTorch DataLoader during iteration on data. Defaults to 0.
            allow_existing: Whether to allow using an existing table. Defaults to True.
            distribute: Whether to use distributed weighing. Not implemented yet. Defaults to False.
            drop_table: Whether to drop the table after dataset creation when using
                `weigh_in_dataset`. Defaults to False.
            max_batches: Optional maximum number of batches to process. If None, all batches are ingested.
            random_state: Optional random state for reproducibility during chunk assignment. Default is None.
                The seed is reused for each new chunk assignment.

        Raises:
            ValueError: If `chunk` or `weighing_batch_size` is not a positive integer
                or `None`, or column names collide with identity columns or each
                other as checked by `GoldDoerWithTable`.
            NotImplementedError: If `distribute` is True.
        """
        self.weigher = weigher
        validate_positive_integer(
            weighing_batch_size,
            "weighing_batch_size",
            allow_zero=False,
            allow_none=True,
        )
        self.weighing_batch_size = weighing_batch_size
        self.reducer = reducer
        validate_positive_integer(chunk, "chunk", allow_none=True)
        self.chunk = chunk
        self.vectorized_key = vectorized_key
        self.include_vectorized_in_table = include_vectorized_in_table
        self.weight_key = weight_key
        self.label_key = label_key
        if distribute:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldDensityWeigher."
            )
        self._distribute = distribute
        self.random_state = random_state

        super().__init__(
            table_path=table_path,
            label_key=label_key,
            allow_existing=allow_existing,
            drop_table=drop_table,
            max_batches=max_batches,
            collate_fn=collate_fn,
            to_keep_schema=to_keep_schema,
            min_pxt_insert_size=min_pxt_insert_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @property
    def distribute(self) -> bool:
        """Whether distributed processing is enabled."""
        return self._distribute

    @distribute.setter
    def distribute(self, value: bool) -> None:
        """Set whether distributed processing is enabled.

        Raises:
            NotImplementedError: If `value` is True.
        """
        if value:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldDensityWeigher."
            )
        self._distribute = False

    def weigh_in_table(self, to_weigh: Dataset | Table) -> Table:
        """Weigh vectorized samples using density weighing and store results in a PixelTable table.

        The weighing process applies a density weighing algorithm on already vectorized
        representations of the data points and stores results in a PixelTable table specified by `table_path`.
        When the chunk attribute is set, the weighing is performed in chunks to reduce memory consumption.
        If a reducer is provided, the vectors are reduced in dimension before applying the density weighing.

        This method is idempotent, meaning that if it is called multiple times on the same
        dataset or table, it will resume the weighing process based on the scores already
        present in the PixelTable table. Inputs and scoring settings must remain unchanged.
        The `max_batches` attribute limits the source prefix processed on each call;
        later calls can add new rows without changing existing raw scores.

        Args:
            to_weigh: Dataset or Table to weigh. If a Dataset is provided, each item should be a
                dictionary with at least the `vectorized_key` key after applying the collate_fn.
                Each vector must be a finite floating one-dimensional tensor. Missing `idx_vector`
                values are generated from item positions, and missing `idx` values default to those vector IDs.
                If a Table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_vector` columns.
                Vector indices must be unique; multiple vectors may belong to the same sample `idx`.

        Returns:
            A PixelTable Table containing at least the weighing information in the `weight_key` column
            and `idx` (index of the sample) and `idx_vector` (index of the vector) columns as well.
            The table also contains the vector column when available and configured retained columns.

        Raises:
            ValueError: If `allow_existing` is False and the result table exists,
                required result columns or its primary key are missing, the source
                vector column is missing, reducer or scoring outputs are malformed, or
                same-sample exclusion leaves a query with fewer than `k` eligible
                references. Other invalid
                inputs may raise errors from collation, tensor operations, or PixelTable.
        """
        weighing_table = self.setup_weighing_table(to_weigh)

        if self.distribute:
            self._distributed_weigh(
                weighing_table,
                to_weigh=to_weigh if isinstance(to_weigh, Table) else weighing_table,
            )
        else:
            self._sequential_weigh(
                weighing_table,
                to_weigh=to_weigh if isinstance(to_weigh, Table) else weighing_table,
            )
        return weighing_table

    def weigh_in_dataset(self, to_weigh: Dataset | Table) -> GoldPxtTorchDataset:
        """Weigh vectorized samples using density weighing and return results as a GoldPxtTorchDataset.

        The weighing process applies a density weighing algorithm on already vectorized
        representations of the data points and stores results in a PixelTable table specified by `table_path`.
        When the chunk attribute is set, the weighing is performed in chunks to reduce memory consumption.
        If a reducer is provided, the vectors are reduced in dimension before applying the density weighing.

        This is a convenience wrapper that runs `weigh_in_table` to populate
        (or resume populating) the PixelTable table, then wraps the results into a
        `GoldPxtTorchDataset` for downstream consumption. If `drop_table` is True,
        the table will be removed after the dataset is created. The returned
        cached dataset remains usable after deletion. Scores remain raw per-vector
        values; use `get_weights_per_idx` for sample aggregation or normalization.

        Args:
            to_weigh: Dataset or Table to weigh. If a Dataset is provided, each item should be a
                dictionary with at least the `vectorized_key` key after applying the collate_fn.
                Each vector must be a finite floating one-dimensional tensor. Missing `idx_vector`
                values are generated from item positions, and missing `idx` values default to those vector IDs.
                If a Table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_vector` columns.
                Vector indices must be unique; multiple vectors may belong to the same sample `idx`.

        Returns:
            A GoldPxtTorchDataset containing at least the weighing information in the `weight_key` key
            and `idx` (index of the sample) and `idx_vector` (index of the vector) keys as well.

        Raises:
            ValueError: Under the validation conditions documented by `weigh_in_table`.
        """
        weighed_table = self.weigh_in_table(to_weigh)

        dataset = GoldPxtTorchDataset(weighed_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(self.table_path)

        return dataset

    def setup_weighing_table(self, to_weigh: Dataset | Table) -> Table:
        """Set up the weighing table based on the source dataset or table.

        Args:
            to_weigh: Dataset or Table to weigh. If a Dataset is provided, each item should be a
                dictionary with at least the `vectorized_key` key after applying the collate_fn.
                If a Table is provided, it should contain at least the `vectorized_key`,
                `idx` and `idx_vector` columns.

        Returns:
            A PixelTable Table with the appropriate schema for weighing,
            populated with rows corresponding to the source dataset or table up to `max_batches`.
            Existing rows and scores are preserved. New rows retain supplied scores;
            otherwise their scores are initialized to null. No scoring is performed.

        Raises:
            ValueError: If `allow_existing` is False and the table at `table_path` already exists,
                if an existing weighing table is missing the expected primary key `idx_vector`,
                or if required result columns or the source vector column are missing.
                This does not validate all stored column types or source values.
        """
        logger.info(f"Loading the existing weighing table from {self.table_path}")
        try:
            old_weighing_table = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            logger.info(f"No existing weighing table from {self.table_path}")
            old_weighing_table = None

        if old_weighing_table is not None:
            check_pxt_table_has_primary_key(old_weighing_table, {"idx_vector"})

        if not self.allow_existing and old_weighing_table is not None:
            raise ValueError(
                f"Table at path {self.table_path} already exists while allow_existing is False."
            )

        if isinstance(to_weigh, Table):
            weighing_table = self._weighing_table_from_table(
                weigh_from=to_weigh,
                old_weighing_table=old_weighing_table,
            )
        else:
            weighing_table = self._weighing_table_from_dataset(
                to_weigh, old_weighing_table
            )

        return weighing_table

    def _weighing_table_from_table(
        self, weigh_from: Table, old_weighing_table: Table | None
    ) -> Table:
        """Create or validate the weighing table schema from a PixelTable table.

        This private method sets up the table structure with necessary columns for tracking
        weighing status and ingests the source prefix allowed by `max_batches`.
        Existing rows are retained, and supplied source scores are copied for new rows.

        Args:
            weigh_from: The source PixelTable table to weigh.
            old_weighing_table: Existing weighing table if resuming, or None.

        Returns:
            The weighing table with proper schema and initial rows.

        Raises:
            ValueError: If the source vector column or required result columns are
                missing, or an empty source cannot supply a vector shape when
                `include_vectorized_in_table` is True. Column types and all source
                values are not exhaustively validated here.
        """
        minimal_schema = self._MINIMAL_SCHEMA.copy()

        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.label_key is not None:
            minimal_schema[self.label_key] = pxt.String

        weighing_table = get_valid_table(
            table=old_weighing_table
            if old_weighing_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )

        if self.vectorized_key not in weigh_from.columns():
            raise ValueError(
                f"Source table does not contain vector column '{self.vectorized_key}'."
            )

        weighing_table_cols = weighing_table.columns()
        if self.weight_key not in weighing_table_cols:
            weighing_table.add_column(if_exists="error", **{self.weight_key: pxt.Float})

        if (
            self.include_vectorized_in_table
            and self.vectorized_key not in weighing_table_cols
        ):
            idx_vectors = [row["idx_vector"] for row in weigh_from.sample(1).collect()]
            if not idx_vectors:
                raise ValueError(
                    f"Source table at {self.table_path} is empty, cannot sample vectorized data for schema inference."
                )
            sample = get_sample_row_from_idx(
                weigh_from,
                idx_key="idx_vector",
                # make sure to take an existing vector
                idx=idx_vectors[0],
                collate_fn=self.collate_fn,
                expected_keys=[self.vectorized_key],
            )

            vectorized_value = sample[self.vectorized_key].squeeze(0)
            weighing_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized_value.shape, pxt.Float
                    ]
                }
            )

        if weighing_table.count() > 0:
            to_select_indices = set(
                [
                    row["idx_vector"]
                    for row in weigh_from.select(weigh_from.idx_vector)
                    .distinct()
                    .collect()
                ]
            )
            already_in_weighing = set(
                [
                    row["idx_vector"]
                    for row in weighing_table.select(weighing_table.idx_vector)
                    .distinct()
                    .collect()
                ]
            )
            still_to_weigh = to_select_indices.difference(already_in_weighing)
            if not still_to_weigh:
                logger.info(
                    f"The weighing table is already initialized in {self.table_path}"
                )
                return weighing_table

        col_list = ["idx", self.vectorized_key]
        weigh_from_cols = weigh_from.columns()
        if self.to_keep_schema is not None:
            col_list.extend(list(self.to_keep_schema.keys()))
        if self.weight_key in weigh_from_cols:
            col_list.append(self.weight_key)
        if self.label_key is not None and self.label_key in weigh_from_cols:
            col_list.append(self.label_key)

        self._add_rows_to_weighing_table_from_dataset(
            weigh_from=GoldPxtTorchDataset(
                weigh_from.select(
                    *[
                        get_expr_from_column_name(weigh_from, col)
                        for col in dict.fromkeys(col_list + ["idx_vector"])
                    ]
                ),
                keep_cache=False,
            ),
            weighing_table=weighing_table,
            include_vectorized=self.include_vectorized_in_table,
        )

        return weighing_table

    def _weighing_table_from_dataset(
        self, weigh_from: Dataset, old_weighing_table: Table | None
    ) -> Table:
        """Create or validate the weighing table schema from a PyTorch Dataset.

        This private method sets up the table structure with necessary columns
        and validates that the vectorized key is present in the dataset.

        Args:
            weigh_from: The source PyTorch Dataset to weigh.
            old_weighing_table: Existing weighing table if resuming, or None.

        Returns:
            The weighing table with proper schema and initial rows.

        Raises:
            ValueError: If required result columns are missing, or schema inference
                finds a non-dictionary sample or a sample missing the vectorized key.
                Source values and stored column types are not exhaustively validated here.
        """
        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.label_key is not None:
            minimal_schema[self.label_key] = pxt.String

        weighing_table = get_valid_table(
            table=old_weighing_table
            if old_weighing_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )
        weighing_table_cols = weighing_table.columns()
        if self.vectorized_key not in weighing_table_cols:
            sample = get_dataset_sample_dict(
                weigh_from,
                collate_fn=self.collate_fn,
                expected=[self.vectorized_key],
            )

            vectorized_value = (
                sample[self.vectorized_key].squeeze(0).detach().cpu().numpy()
            )
            weighing_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        tuple(vectorized_value.shape), pxt.Float
                    ]
                }
            )

        if self.weight_key not in weighing_table_cols:
            weighing_table.add_column(if_exists="error", **{self.weight_key: pxt.Float})

        self._add_rows_to_weighing_table_from_dataset(
            weigh_from,
            weighing_table,
            include_vectorized=True,
        )

        return weighing_table

    def _add_rows_to_weighing_table_from_dataset(
        self,
        weigh_from: Dataset,
        weighing_table: Table,
        include_vectorized: bool = False,
    ) -> None:
        """Add rows from the source dataset to the weighing table.

        This private method iterates through the dataset in batches and populates the
        weighing table with vectorized data and metadata, skipping already processed vectors.
        The `max_batches` attribute limits the number of source batches processed on each call.
        Missing vector indices are generated from item positions; missing sample indices
        default to the vector indices.

        Args:
            weigh_from: The source PyTorch Dataset.
            weighing_table: The weighing table to populate.
            include_vectorized: Whether to include the vectorized data in the weighing table.

        Raises:
            ValueError: If a batch is missing `vectorized_key`, contains duplicate vector indices,
                or contains vectors that are not finite floating one-dimensional tensors.
        """
        dataloader = DataLoader(
            weigh_from,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        # the vectors already present in the weighing table will be excluded
        # from the batches to process to avoid duplicates and speed up processing
        to_exclude_from_batch = {
            int(row["idx_vector"])
            for row in weighing_table.select(weighing_table.idx_vector).collect()
        }

        ready_to_insert: list[dict[str, Any]] = []

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Initializing rows for the weighing table",
            total=(
                None if hasattr(weigh_from, "__len__") is False else len(dataloader)
            ),
        ):
            # Stop if we've processed enough batches
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            if "idx_vector" not in batch:
                starts = batch_idx * self.batch_size
                batch["idx_vector"] = [
                    starts + idx for idx in range(len(batch[self.vectorized_key]))
                ]

            if "idx" not in batch:
                batch["idx"] = batch["idx_vector"]

            # Remove the vectors already present in the weighing table to avoid duplicates and speed up processing
            if to_exclude_from_batch:
                batch = filter_batch_from_indices(
                    batch,
                    to_exclude_from_batch,
                    index_key="idx_vector",
                )

                if len(batch) == 0:
                    continue  # all samples already described

            if self.weight_key not in batch:
                batch[self.weight_key] = [
                    None for _ in range(len(batch[self.vectorized_key]))
                ]

            # add newly added vectors to the exclusion set to avoid duplicates in the next batches
            to_exclude_from_batch.update(
                [
                    int(idx.item() if isinstance(idx, torch.Tensor) else idx)
                    for idx in batch["idx_vector"]
                ]
            )
            to_insert_keys = ["idx", self.weight_key]
            if self.to_keep_schema is not None:
                to_insert_keys.extend(list(self.to_keep_schema.keys()))
            if self.label_key is not None and self.label_key in batch:
                to_insert_keys.append(self.label_key)
            if include_vectorized:
                to_insert_keys.append(self.vectorized_key)
                batch[self.vectorized_key] = (
                    batch[self.vectorized_key].detach().cpu().numpy()
                )

            batch_as_list = make_batch_ready_for_table(
                batch,
                to_insert_keys,
                "idx_vector",
            )

            ready_to_insert.extend(batch_as_list)
            if len(ready_to_insert) >= self.min_pxt_insert_size:
                weighing_table.insert(ready_to_insert)
                ready_to_insert = []

        if ready_to_insert:
            weighing_table.insert(ready_to_insert)

    def _distributed_weigh(self, weighing_table: Table, to_weigh: Table) -> None:
        """Reject distributed weighing, which is not implemented.

        Args:
            weighing_table: The table in which to store weighing results.
            to_weigh: The source table containing vectorized data.

        Raises:
            NotImplementedError: Always raised as distributed mode is not yet implemented.
        """
        raise NotImplementedError("Distributed weighing is not implemented yet.")

    def _sequential_weigh(self, weighing_table: Table, to_weigh: Table) -> None:
        """Sequentially weigh each label independently, or all vectors when no label is configured.

        Args:
            weighing_table: The table in which to store weighing results.
            to_weigh: The source table containing vectorized data.
        """
        if self.label_key is not None:
            label_col = get_expr_from_column_name(weighing_table, self.label_key)
            labels = sorted(
                {
                    row[self.label_key]
                    for row in weighing_table.select(label_col).distinct().collect()
                }
            )
            for label_value in labels:
                logger.info(f"Weighing vectors for label '{label_value}'")
                self._weigh_label(
                    weigh_from=to_weigh,
                    weighing_table=weighing_table,
                    label_value=label_value,
                )
        else:
            logger.info("Weighing all vectors")
            self._weigh_label(weigh_from=to_weigh, weighing_table=weighing_table)

    def _weigh_label(
        self,
        weigh_from: Table,
        weighing_table: Table,
        label_value: str | None = None,
    ) -> None:
        """Perform chunked density weighing for a specific label or all data.

        Pending vectors are processed in chunks with optional dimensionality
        reduction, and scores are stored by vector index.

        Args:
            weigh_from: The source table with vectorized data.
            weighing_table: The table to store weighing results.
            label_value: Optional label value to filter vectors by.

        Raises:
            ValueError: If a default kNN chunk has fewer than `k + 2` vectors, or
                the reducer or weighing tool returns invalid results. Same-sample
                exclusion instead reports a query that lacks `k` eligible references.
        """
        weight_col = get_expr_from_column_name(weighing_table, self.weight_key)

        vector_table = (
            weighing_table if self.include_vectorized_in_table else weigh_from
        )
        vectorized_col = get_expr_from_column_name(vector_table, self.vectorized_key)

        # do not recompute weights for already processed vectors
        available_query = weight_col == None  # noqa: E711
        label_scope = ""
        if label_value is not None:
            assert self.label_key is not None
            label_col = get_expr_from_column_name(weighing_table, self.label_key)
            available_query = available_query & (label_col == label_value)
            label_scope = f" for label '{label_value}'"

        weighing_pool = weighing_table.where(available_query)

        # when all vectors are already computed skip the label
        if weighing_pool.count() == 0:
            return

        logger.info(f"Starting density weighing{label_scope}")
        to_weigh_vector_indices = [
            int(row["idx_vector"])
            for row in weighing_pool.select(weighing_table.idx_vector).collect()
        ]
        chunk_assignment = get_random_chunk_assignment(
            to_chunk=sorted(to_weigh_vector_indices),
            max_chunk_size=self.chunk,
            random_state=self.random_state,
        )
        knn_tool = (
            self.weigher
            if isinstance(self.weigher, GoldKnnDensityWeighingTool)
            else None
        )
        if (
            knn_tool is not None
            and not knn_tool.exclude_same_sample
            and any(len(indices) < knn_tool.k + 2 for indices in chunk_assignment)
        ):
            raise ValueError("Each weighing chunk must contain at least k + 2 vectors.")
        for chunk_id, chunk_vector_indices in enumerate(chunk_assignment):
            logger.info(
                f"Weighing {len(chunk_vector_indices)} vectors{label_scope} "
                f"from chunk {chunk_id}"
            )

            to_weigh_from = vector_table.where(
                vector_table.idx_vector.isin(chunk_vector_indices)
            ).select(vector_table.idx_vector, vectorized_col)
            to_weigh_for_chunk = to_weigh_from.collect()

            indices = [int(row["idx_vector"]) for row in to_weigh_for_chunk]
            vectors = torch.stack(
                [
                    torch.as_tensor(row[self.vectorized_key])
                    for row in to_weigh_for_chunk
                ],
                dim=0,
            )
            sample_idx = (
                self._sample_idx_for_vectors(weighing_table, indices)
                if knn_tool is not None and knn_tool.exclude_same_sample
                else None
            )
            scores = self._compute_weights(vectors, sample_idx)
            weighing_table.batch_update(
                [
                    {"idx_vector": idx, self.weight_key: float(score.item())}
                    for idx, score in zip(indices, scores)
                ]
            )
            logger.info(
                f"Weighed {len(indices)} vectors after the current chunk{label_scope}"
            )

    def _sample_idx_for_vectors(
        self, weighing_table: Table, idx_vectors: list[int]
    ) -> torch.Tensor:
        """Return sample ids in the same order as `idx_vectors`.

        Args:
            weighing_table: Table whose `idx` and `idx_vector` columns associate
                each vector with its source sample.
            idx_vectors: Vector ids in the row order of the vectors about to be
                scored.

        Returns:
            An int64 tensor shaped `(len(idx_vectors),)`.

        Raises:
            ValueError: If the weighing table has no sample idx for a vector id.
        """
        rows = (
            weighing_table.where(weighing_table.idx_vector.isin(idx_vectors))
            .select(weighing_table.idx_vector, weighing_table.idx)
            .collect()
        )
        idx_by_vector = {int(row["idx_vector"]): int(row["idx"]) for row in rows}
        missing = [idx for idx in idx_vectors if idx not in idx_by_vector]
        if missing:
            raise ValueError(
                f"Weighing table is missing sample idx for vector indices {missing}."
            )
        return torch.tensor(
            [idx_by_vector[idx] for idx in idx_vectors], dtype=torch.long
        )

    def _compute_weights(
        self, vectors: torch.Tensor, sample_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the weighing algorithm.

        If a reducer is provided, the vectors are reduced before scoring. A
        `GoldReductionToolWithFit` is applied with `fit_transform`; a transform-only
        reducer is applied with `transform`. Each scoring batch uses the full
        reduced chunk as its reference population. `sample_idx` stays in the input
        row order. The reducer must keep that order; only the row count is checked.

        Args:
            vectors: Input vectors to weigh, shaped `(num_vectors, feature_dim)`.
            sample_idx: Optional sample id per input row, required by a kNN tool
                that excludes same-sample neighbors. Split with the query batches.

        Returns:
            A tensor shaped `(num_vectors,)` containing weighing score
            per input vector, in the same row order.

        Raises:
            ValueError: If the reducer does not return a two-dimensional tensor with the
                same number of rows, the weighing tool does not return one finite
                nonnegative score per input vector, or same-sample exclusion is
                missing sample ids.
        """
        expected_rows = len(vectors)

        if self.reducer is not None:
            vectors = (
                self.reducer.fit_transform(vectors)
                if isinstance(self.reducer, GoldReductionToolWithFit)
                else self.reducer.transform(vectors)
            )

        if (
            not isinstance(vectors, torch.Tensor)
            or vectors.ndim != 2
            or len(vectors) != expected_rows
        ):
            raise ValueError(
                "Reducer output must be 2D and preserve the number of rows."
            )
        knn_tool = (
            self.weigher
            if isinstance(self.weigher, GoldKnnDensityWeighingTool)
            and self.weigher.exclude_same_sample
            else None
        )
        if self.weighing_batch_size is None:
            batches = [vectors]
            query_ids: list[torch.Tensor | None] = [sample_idx]
        else:
            batches = list(vectors.split(self.weighing_batch_size))
            query_ids = (
                list(sample_idx.split(self.weighing_batch_size))
                if sample_idx is not None
                else [None] * len(batches)
            )
        results = []
        for batch, batch_ids in zip(batches, query_ids, strict=True):
            if knn_tool is not None:
                if sample_idx is None or batch_ids is None:
                    raise ValueError(
                        "sample_idx is required when exclude_same_sample is True."
                    )
                scores = (
                    knn_tool.weigh(vectors, sample_idx=sample_idx)
                    if self.weighing_batch_size is None
                    else knn_tool.weigh_batch(
                        vectors,
                        batch,
                        sample_idx=sample_idx,
                        query_sample_idx=batch_ids,
                    )
                )
            else:
                scores = (
                    self.weigher.weigh(vectors)
                    if self.weighing_batch_size is None
                    else self.weigher.weigh_batch(vectors, batch)
                )
            if (
                not isinstance(scores, torch.Tensor)
                or scores.shape != (len(batch),)
                or not torch.isfinite(scores).all()
                or (scores < 0).any()
            ):
                raise ValueError(
                    "Weigher output must contain one finite nonnegative score per vector."
                )
            results.append(scores)
        return torch.cat(results)

    @staticmethod
    def get_weights_per_idx(
        table: Table,
        weight_key: str = "weight",
        idx_key: Literal["idx", "idx_vector"] = "idx",
        label_key: str | None = None,
        label_value: str | None = None,
        aggregation: Callable[[list[float]], float] = mean,
        normalize: bool = True,
        max_raw_score: float | None = None,
    ) -> dict[int, float]:
        """Retrieve weights by sample or vector index without changing stored scores.

        Sample scores are aggregated, then optionally clipped, then normalized.
        By default, each returned score is divided by the largest returned score
        when it is positive; all-zero scores remain zero. The default
        `statistics.mean` avoids overflow when aggregating large finite scores.

        Args:
            table: PixelTable table in which every vector has a score.
            weight_key: Column containing raw vector scores. Defaults to "weight".
            idx_key: "idx" aggregates vectors belonging to each sample;
                "idx_vector" returns individual vector scores. Defaults to "idx".
            label_key: Optional label column to filter. Requires `label_value`.
            label_value: Label to select when `label_key` is provided.
            aggregation: Function applied to each sample's vector scores.
                Defaults to the arithmetic mean. Ignored for "idx_vector".
            normalize: Whether to divide by the maximum returned score after
                filtering and aggregation. Defaults to True. False returns raw
                or aggregated scores without rescaling.
            max_raw_score: Optional maximum for the returned raw scores. Scores
                above it are set to it and smaller scores are unchanged. It is
                applied after sample aggregation (or directly to vector scores
                for "idx_vector") and before normalization, so the clipped
                maximum becomes the normalization reference. Must be finite and
                positive. Defaults to None, which disables clipping.

        Returns:
            A dictionary mapping the requested sample or vector indices to weights.

        Raises:
            ValueError: If `label_key` is provided without `label_value`, or if
                `max_raw_score` is not a finite positive number.
            RuntimeError: If any vector in the table lacks a score, or no scores
                are found for the requested filter.
        """
        validate_positive_number(max_raw_score, "max_raw_score", allow_none=True)

        weight_col = get_expr_from_column_name(table, weight_key)
        if table.where(weight_col == None).count() > 0:  # noqa: E711
            raise RuntimeError(
                "Not all vectors have scores in the table. "
                "Cannot compute weights per sample index."
            )

        idx_vector_col = get_expr_from_column_name(table, "idx_vector")
        idx_col = get_expr_from_column_name(table, "idx")
        pool: Table | Query = table
        if label_key is not None:
            if label_value is None:
                raise ValueError(
                    "If label_key is provided, label_value must also be provided."
                )
            label_col = get_expr_from_column_name(table, label_key)
            pool = pool.where(label_col == label_value)

        rows = pool.select(idx_col, idx_vector_col, weight_col).collect()
        scores: dict[int, float]
        if idx_key == "idx_vector":
            scores = {int(row["idx_vector"]): float(row[weight_key]) for row in rows}
        else:
            scores_per_index = defaultdict(list)
            for row in rows:
                scores_per_index[int(row["idx"])].append(float(row[weight_key]))

            scores = {}
            for idx, index_scores in scores_per_index.items():
                scores[idx] = aggregation(index_scores)

        if not scores:
            raise RuntimeError(
                "No scores found for the specified index key and label filter."
            )

        if max_raw_score is not None:
            scores = {idx: min(score, max_raw_score) for idx, score in scores.items()}

        if normalize:
            max_score = np.max(list(scores.values()))
            if max_score > 0:
                scores = {idx: score / max_score for idx, score in scores.items()}

        return scores
