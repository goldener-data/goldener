"""Clustering of vectorized samples.

Clustering groups the vectorized samples in clusters, gathering similar content unless a random clustering is
used. The clusters are used to keep the variability of the data
when selecting or splitting it, and to balance the batches during training (`goldener.organize`).

- `GoldClusteringTool` is the abstract interface of the clustering algorithms.
- `GoldSKLearnClusteringTool` wraps a scikit-learn clustering estimator providing `predict` and `n_clusters`.
- `GoldRandomClusteringTool` chunks the data randomly into clusters of almost equal size.
- `GoldClusterizer` runs the clustering on a whole dataset of vectors, possibly in chunks, and stores the cluster
  assignments locally (one per vector by default, a sample with multiple vectors can then have multiple clusters).
  A companion view persists chunk membership for resumed clustering.
"""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from logging import getLogger
from typing import Callable, Any

import numpy as np
import pixeltable as pxt
from pixeltable import Error, Query
from pixeltable.catalog import Table

import torch
from sklearn.base import ClusterMixin
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from goldener.do import GoldDoerWithTable
from goldener.pxt_utils import (
    set_value_to_idx_rows,
    GoldPxtTorchDataset,
    get_expr_from_column_name,
    get_valid_table,
    make_batch_ready_for_table,
    check_pxt_table_has_primary_key,
    get_sample_row_from_idx,
    assign_chunk,
    get_chunk_view,
)
from goldener.reduce import GoldReductionTool, GoldReductionToolWithFit
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import (
    validate_positive_integer,
    validate_vectorized_input,
    filter_batch_from_indices,
)


logger = getLogger(__name__)


class GoldClusteringTool(ABC):
    """Run clustering on input vectors."""

    @abstractmethod
    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """Fit the clustering tool from a given set of vectors into n clusters.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)` with
                positive feature width. Each row represents one vector.
            n_clusters: Number of clusters to form.

        Returns:
            A 1D tensor with one cluster index per input row.
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Assign the input vectors to the clusters.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)` with
                positive feature width. Each row represents one vector.

        Returns:
            A 1D tensor with one cluster index per input row.
        """
        pass


class GoldRandomClusteringTool(GoldClusteringTool):
    """Chunk data randomly into clusters of almost equal size.

    A successful `fit` labels row `i` with `i % n_clusters`, then shuffles those
    labels with this tool's NumPy `RandomState`. For `num_rows` rows, cluster
    sizes differ by at most one. When `num_rows % n_clusters` is `r` and `r` is
    not zero, cluster ids `0` through `r - 1` each receive one more row than the
    other ids. The shuffle changes which rows receive which ids and leaves those
    sizes unchanged.

    The tool always owns a `RandomState`. An integer seeds a new one. None draws
    entropy. A passed `RandomState` is used as that instance. Each successful
    `fit` of one or more rows consumes it, so a later `fit` on the same instance
    does not repeat the shuffle drawn from the original state.

    `predict` is not supported. A fit does not store centroids or another mapping
    that could label new rows.
    """

    def __init__(self, random_state: np.random.RandomState | int | None = None) -> None:
        """Build the NumPy `RandomState` consumed by later fits.

        Args:
            random_state: Seed or random state for the assignments. An integer
                seeds a new `RandomState`. None draws entropy. A `RandomState`
                is used as this tool's own random state.

        Raises:
            ValueError: If `random_state` is not a `RandomState`, an integer,
                or None.
        """
        if random_state is not None and not isinstance(
            random_state, int | np.random.RandomState
        ):
            raise ValueError(
                "The random state must be a NumPy RandomState, an integer or None, "
                f"got {type(random_state).__name__}."
            )
        if isinstance(random_state, int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        self._random_state = random_state

    @property
    def random_state(self) -> dict[str, Any]:
        """Return a snapshot of this tool's random state.

        Reading this property does not advance the random state. The result is
        the state returned by `RandomState.get_state(legacy=False)`. It is not
        the seed passed to `__init__`.

        Returns:
            A snapshot of the current `RandomState`.
        """
        return self._random_state.get_state(legacy=False)

    def update_random_state(self, state: dict[str, Any]) -> None:
        """Restore this tool's random state from a previous snapshot.

        The next successful `fit` uses the restored state.

        Args:
            state: Random state, as returned by `random_state`.

        Raises:
            TypeError: If `state` is not a dictionary or a tuple.
            ValueError: If `state` is not a valid `RandomState` state.
        """
        self._random_state.set_state(state)

    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """Randomly assign each input vector to clusters of roughly the same size.

        Labels row `i` with `i % n_clusters`, then shuffles those labels with
        this tool's NumPy `RandomState`. A one-row fit does not call `shuffle`.
        `RandomState.shuffle` draws nothing for a one-element sequence, so that
        fit draws one value with `RandomState.random()`. A successful call of
        one or more rows advances the random state.
        A call that raises does not.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)` with
                positive feature width. Each row represents one vector.
            n_clusters: Number of clusters to form.

        Returns:
            A 1D tensor with one cluster index per input row.

        Raises:
            ValueError: If `x` is not a finite floating 2D tensor with positive
                feature width, or `n_clusters` exceeds the number of rows in `x`.
        """
        validate_vectorized_input(x, "GoldRandomClusteringTool")
        total = len(x)

        if n_clusters > total:
            raise ValueError(
                f"Number of clusters ({n_clusters}) cannot be greater than "
                f"the number of samples ({total})."
            )

        cluster_assignment = [i % n_clusters for i in range(total)]
        if len(cluster_assignment) == 1:
            self._random_state.random()
        else:
            self._random_state.shuffle(cluster_assignment)

        return torch.tensor(
            cluster_assignment,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Random clustering tool does not support predict.")


def get_random_chunk_assignment(
    to_chunk: list[int],
    max_chunk_size: int | None = None,
    random_state: int | None = None,
) -> list[list[int]]:
    """Assign indices to random chunks.

    Args:
        to_chunk: List of indices to chunk.
        max_chunk_size: Maximum size of each chunk. If None, all indices are returned
            in a single chunk.
        random_state: Optional random state for reproducibility during chunk assignment.

    Returns:
        A list of chunks, where each chunk is a list of indices
        from `to_chunk` randomly assigned.

    Raises:
        ValueError: If `to_chunk` is empty or if `max_chunk_size` is not a positive integer or None.
    """
    if not to_chunk:
        raise ValueError("to_chunk must be a non-empty list of indices.")

    validate_positive_integer(max_chunk_size, "max_chunk_size", allow_none=True)

    to_chunk_size = len(to_chunk)
    if max_chunk_size is None or max_chunk_size >= to_chunk_size:
        return [to_chunk]  # all in one chunk

    chunk_count = math.ceil(
        to_chunk_size / max_chunk_size
    )  # make sure chunk size is not above specification
    random_assignment = (
        GoldRandomClusteringTool(random_state=random_state)
        .fit(
            torch.zeros(to_chunk_size, 1),  # dummy input for random clustering
            chunk_count,
        )
        .tolist()
    )

    return [
        [
            idx_vector
            for vect_pos, idx_vector in enumerate(to_chunk)
            if random_assignment[vect_pos] == chunk_idx
        ]
        for chunk_idx in range(chunk_count)
    ]


def get_chunk_assignments_from_table(
    table: Table,
    restrict_to: set[int] | None = None,
    max_chunk_size: int | None = None,
    random_state: int | None = None,
) -> list[list[int]]:
    """Get the chunk assignments from a PixelTable table.

    When all idx_vector have their chunk, they are returned as is.
    When some idx_vector have no chunk, they are randomly assigned to
    chunks of size at most max_chunk_size.

    Args:
        table: The PixelTable table to get the chunk assignments from.
        restrict_to: Optional set of idx_vector values to restrict the chunk assignments to.
        max_chunk_size: Maximum size of each chunk. If None, all indices are returned
            in a single chunk.
        random_state: Optional random state for reproducibility during chunk assignment.

    Returns:
        A list of chunks, where each chunk is a list of indices
        from the table randomly assigned.
    """
    saved_chunks: dict[int | None, list[int]] = defaultdict(list)
    query = (
        table.where(table.idx_vector.isin(restrict_to))
        if restrict_to is not None
        else table
    )

    for row in query.select(table.idx_vector, table.chunk).collect():
        saved_chunks[row["chunk"]].append(row["idx_vector"])

    if None not in saved_chunks:
        return [
            saved_chunks[key]
            for key in sorted([key for key in saved_chunks if key is not None])
        ]

    chunk_assignment = get_random_chunk_assignment(
        to_chunk=sorted(
            [
                idx_vector
                for chunk_indices in saved_chunks.values()
                for idx_vector in chunk_indices
            ]
        ),
        max_chunk_size=max_chunk_size,
        random_state=random_state,
    )
    chunk_by_vector = {
        str(idx_vector): chunk_idx
        for chunk_idx, chunk_indices in enumerate(chunk_assignment)
        for idx_vector in chunk_indices
    }
    table.update(
        {"chunk": assign_chunk(table.idx_vector, table.chunk, chunk_by_vector)}
    )

    return chunk_assignment


class GoldSKLearnClusteringTool(GoldClusteringTool):
    """Cluster data using a wrapped scikit-learn clustering estimator.

    The tool controls the randomness of stochastic estimators (such as KMeans) when
    it holds a NumPy `RandomState`: one given or seeded by `random_state`, one
    already set on the estimator, or one created by `update_random_state`. Its state
    can then be inspected with `random_state` and restored with `update_random_state`
    to replay the same random choices. Otherwise, the estimator randomness is left
    untouched. Estimators without a `random_state` parameter (such as Birch) do not
    support these controls.

    Attributes:
        tool: The wrapped scikit-learn clustering estimator.
    """

    def __init__(
        self,
        tool: ClusterMixin,
        random_state: np.random.RandomState | int | None = None,
    ) -> None:
        """Initialize the GoldSKLearnClusteringTool.

        Args:
            tool: A scikit-learn clustering estimator providing `predict` and `n_clusters`.
            random_state: Optional random state overriding the `random_state` of the estimator.
                An integer is used as seed of a new `RandomState`. When None, a `RandomState`
                already set on the estimator is reused; otherwise the estimator randomness is
                left untouched.

        Raises:
            ValueError: If `tool` has no `predict` method, if `random_state` is not a
                `RandomState`, an integer or None, or if `random_state` is not None
                while `tool` has no `random_state` parameter.
            NotImplementedError: If `tool` does not allow specifying `n_clusters`.
        """
        if not hasattr(tool, "predict"):
            raise ValueError(
                f"The clustering tool {tool} must provide a predict method."
            )

        if not hasattr(tool, "n_clusters"):
            raise NotImplementedError(
                f"The clustering tool {tool} must allow specifying `n_clusters`."
            )

        self.tool = tool

        if random_state is not None:
            if not isinstance(random_state, int | np.random.RandomState):
                raise ValueError(
                    "The random state must be a NumPy RandomState, an integer or None, "
                    f"got {type(random_state).__name__}."
                )
            self._check_random_state_support()

        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        elif random_state is None:
            tool_random_state = getattr(tool, "random_state", None)
            if isinstance(tool_random_state, np.random.RandomState):
                random_state = tool_random_state

        self._random_state = random_state
        if random_state is not None:
            self.tool.random_state = random_state

    def _check_random_state_support(self) -> None:
        """Raise a ValueError if the wrapped estimator has no `random_state` parameter."""
        if not hasattr(self.tool, "random_state"):
            raise ValueError(
                f"The clustering tool {self.tool} has no `random_state` parameter, "
                "its random state cannot be set."
            )

    @property
    def random_state(self) -> dict[str, Any] | None:
        """Return a snapshot of the estimator random state.

        Returns:
            An independent copy of the `RandomState` state, as returned by
            `numpy.random.RandomState.get_state(legacy=False)`, or None when the
            estimator randomness is not controlled by this tool.
        """
        if self._random_state is None:
            return None
        return self._random_state.get_state(legacy=False)

    def update_random_state(self, state: dict[str, Any]) -> None:
        """Restore the estimator random state.

        When the tool has no `RandomState` yet, a new one is created and set on the
        estimator, overriding its previous `random_state`.

        Args:
            state: Random state, as returned by `random_state`.

        Raises:
            ValueError: If the estimator has no `random_state` parameter.
        """
        if self._random_state is None:
            self._check_random_state_support()
            self._random_state = np.random.RandomState()
            self.tool.random_state = self._random_state
        self._random_state.set_state(state)

    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """Fit the scikit-learn clustering tool on the input vectors and return cluster assignments.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)` with
                positive feature width. Each row represents one vector.
            n_clusters: Number of clusters to form.

        Returns:
            A 1D tensor with one cluster index per input row.

        Raises:
            ValueError: If `x` is not a finite floating 2D tensor with positive
                feature width.
        """
        validate_vectorized_input(x, "GoldSKLearnClusteringTool")
        self.tool.n_clusters = n_clusters
        x_np = x.detach().cpu().numpy()
        labels = self.tool.fit(x_np).labels_
        return torch.from_numpy(labels)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Assign the input vectors to clusters using the fitted scikit-learn clustering tool.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)` with
                positive feature width. Each row represents one vector.

        Returns:
            A 1D tensor of cluster indices predicted for each input vector.

        Raises:
            ValueError: If `x` is not a finite floating 2D tensor with positive
                feature width.
        """
        validate_vectorized_input(x, "GoldSKLearnClusteringTool")
        x_np = x.detach().cpu().numpy()
        return torch.from_numpy(self.tool.predict(x_np))


class GoldClusterizer(GoldDoerWithTable):
    """Cluster data points from vectorized samples.

    The GoldClusterizer processes a dataset or PixelTable table to perform clustering using a
    clustering algorithm on already vectorized representations. The clustering results are stored
    in a local PixelTable table (specified by `table_path`) so that the clustering process is idempotent:
    calling the same operation multiple times will not duplicate or recompute clustering that are already
    present in the table.

    If the dataset is too big to fit into memory or the clustering algorithm is too
    computationally expensive, the clustering can be performed in chunks. Chunk membership
    is stored in a view at `<table_path>_chunks`, which adds a nullable integer `chunk`
    column to the intermediate table. Complete assignments for pending vectors are reused
    on resume; incomplete assignments are redrawn in full for the pending population.
    The column name `chunk` is reserved for this view and cannot be used by configurable
    column keys or `to_keep_schema`.
    All torch tensors will be converted to numpy arrays before saving.

    The clustering can operate in a sequential (single-process) mode or a
    distributed mode (not implemented). The table schema is created/validated automatically and will include
    minimal indexing columns (`idx`, `idx_vector`) required to link clustered samples back to
    their originating data points.

    Attributes:
        table_path: Path to the PixelTable table where clustering results will be stored locally.
        clustering_tool: The GoldClusteringTool implementing the clustering algorithm.
        reducer: Optional GoldReductionTool instance for dimensionality reduction before clustering.
        chunk: Optional chunk size for processing data in chunks to reduce memory consumption.
        collate_fn: Optional function to collate dataset samples into batches composed of
            dictionaries with at least the key specified by `vectorized_key` returning a PyTorch Tensor.
            If None, `collate_keeping_sequences_as_sequences` is used, which preserves sequence-valued
            fields as one sequence per sample.
        vectorized_key: Key in the batch dictionary that contains the vectorized data for the clustering. Default is "vectorized".
        include_vectorized_in_table: Whether to keep the vectorized data in the cluster table.
            It is only applied if the cluster table is created from a Table (it is forced anyway for Dataset). Default is False.
        cluster_key: Column name to store the clustering value in the PixelTable table. Default is "cluster".
        label_key: Optional key for label-based stratified clustering.
        to_keep_schema: Optional dictionary defining additional columns to keep from the original dataset/table
            into the cluster table. The keys are the column names and the values are the PixelTable types.
        min_pxt_insert_size: Minimum number of rows to accumulate before inserting into PixelTable. Default is 100.
        batch_size: Batch size used when iterating over the data.
        num_workers: Number of workers for the PyTorch DataLoader during iteration on data.
        allow_existing: If False, an error will be raised when the table already exists. Default is True.
        distribute: Whether to use distributed processing for clustering and table population. Not implemented yet. Default is False.
        drop_table: Whether to drop the cluster table after creating the dataset with clustering results. It is only applied
            when using `cluster_in_dataset`. Default is False.
        max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
        random_state: Optional random state for reproducibility during chunk assignment.
        force_same_cluster: If True, all vectors from the same sample (same `idx`) are assigned
            to the same cluster. The assignment to a cluster depends on the last clusterized vector
            of the sample (suboptimal but easiest solution). Default is False.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
        "idx_vector": pxt.Required[pxt.Int],
    }
    _RESERVED_KEYS: frozenset[str] = frozenset({"idx", "idx_vector", "chunk"})

    def __init__(
        self,
        table_path: str,
        clustering_tool: GoldClusteringTool,
        reducer: GoldReductionTool | None = None,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        vectorized_key: str = "vectorized",
        include_vectorized_in_table: bool = False,
        cluster_key: str = "cluster",
        label_key: str | None = None,
        to_keep_schema: dict[str, type] | None = None,
        min_pxt_insert_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 0,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
        random_state: int | None = None,
        force_same_cluster: bool = False,
    ) -> None:
        """Initialize the GoldClusterizer with configuration parameters.

        Args:
            table_path: Path to the PixelTable table where clustering results will be stored locally.
            clustering_tool: The GoldClusteringTool implementing the clustering algorithm.
            reducer: Optional GoldReductionTool instance for dimensionality reduction before clustering.
            chunk: Optional chunk size for processing data in chunks to reduce memory consumption.
            collate_fn: Optional function to collate dataset samples into batches composed of
                dictionaries with at least the key specified by `vectorized_key` returning a PyTorch Tensor.
                If None, `collate_keeping_sequences_as_sequences` is used, which preserves sequence-valued
                fields as one sequence per sample.
            vectorized_key: Key in the batch dictionary that contains the vectorized data for the clustering. Default is "vectorized".
            include_vectorized_in_table: Whether to keep the vectorized data in the cluster table. Defaults to False.
                    It is only applied if the cluster table is created from a Table (it is forced anyway for Dataset).
            cluster_key: Column name to store the clustering value in the PixelTable table. Default is "cluster".
            label_key: Optional key for label-based stratified clustering.
            to_keep_schema: Optional dictionary defining additional columns to keep from the original dataset/table
                into the cluster table. The keys are the column names and the values are the PixelTable types.
            min_pxt_insert_size: Minimum number of rows to accumulate before inserting into PixelTable. Default is 100.
            batch_size: Batch size used when iterating over the data.
            num_workers: Number of workers for the PyTorch DataLoader during iteration on data.
            allow_existing: If False, an error will be raised when the table already exists. Default is True.
            distribute: Whether to use distributed processing for clustering and table population. Not implemented yet. Default is False.
            drop_table: Whether to drop the cluster table after creating the dataset with clustering results. It is only applied
                when using `cluster_in_dataset`. Default is False.
            max_batches: Optional maximum number of batches to process. Useful for testing on a small subset of the dataset.
            random_state: Optional random state for reproducibility during chunk assignment.
            force_same_cluster: If True, all vectors from the same sample (same `idx`) are assigned
                to the same cluster. The assignment to a cluster depends on the last clusterized vector
                of the sample (suboptimal but easiest solution). Default is False.

        Raises:
            ValueError: If `chunk` is not a positive integer or None, or a configurable
                column key or `to_keep_schema` uses the reserved name `chunk`.
            NotImplementedError: If `distribute` is True.
            ValueError: If a customizable key uses `idx` or `idx_vector`, collides
                with an identity column, or shares its value with another key, or
                if `to_keep_schema` uses one of those columns (see `GoldDoerWithTable`).
        """
        self.clustering_tool = clustering_tool
        self.reducer = reducer
        validate_positive_integer(chunk, "chunk", allow_none=True)
        self.chunk = chunk
        self.vectorized_key = vectorized_key
        self.include_vectorized_in_table = include_vectorized_in_table
        self.cluster_key = cluster_key
        if distribute:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldClusterizer."
            )
        self._distribute = distribute
        self.random_state = random_state
        self.force_same_cluster = force_same_cluster
        super().__init__(
            allow_existing=allow_existing,
            drop_table=drop_table,
            max_batches=max_batches,
            collate_fn=collate_fn,
            to_keep_schema=to_keep_schema,
            min_pxt_insert_size=min_pxt_insert_size,
            batch_size=batch_size,
            num_workers=num_workers,
            table_path=table_path,
            label_key=label_key,
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
                "Distributed processing is not implemented for GoldClusterizer."
            )
        self._distribute = value

    def cluster_in_dataset(
        self,
        cluster_from: Dataset | Table,
        n_clusters: int,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> GoldPxtTorchDataset:
        """Cluster the data and return results as a GoldPxtTorchDataset.

        The clustering process applies a clustering algorithm on already vectorized
        representations of the data points and stores results in a PixelTable table specified by `table_path`.
        When the chunk attribute is set, the clustering is performed in chunks to reduce memory consumption.
        If a reducer is provided, the vectors are reduced in dimension before applying the clustering.

        This is a convenience wrapper that runs `cluster_in_table` to populate
        (or resume populating) the PixelTable table, then wraps the table into a
        `GoldPxtTorchDataset` for downstream consumption. If `drop_table` is True,
        the table will be removed after the dataset is created.

        Args:
            cluster_from: Dataset or Table to cluster. If a Dataset is provided, each item should be a
                dictionary with at least the `vectorized_key`, `idx_vector` and `idx` keys after applying the collate_fn.
                If a Table is provided, it should contain at least the `vectorized_key` and `idx` columns.
            n_clusters: Number of clusters to create.
            restrict_to: Optional set of indices to restrict the clustering to.
                If provided, only the selected indices in `restriction_idx_key` will be used for the clustering.
            restriction_idx_key: Column name used to get sample indices for restriction.
        Returns:
            A GoldPxtTorchDataset containing at least the clustering information in the `cluster_key` key
                and `idx` (index of the sample) and `idx_vector` (index of the vector) keys as well.
        """

        cluster_table = self.cluster_in_table(
            cluster_from,
            n_clusters,
            restrict_to=restrict_to,
            restriction_idx_key=restriction_idx_key,
        )

        cluster_dataset = GoldPxtTorchDataset(cluster_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(f"{self.table_path}_chunks", if_not_exists="ignore")
            pxt.drop_table(cluster_table)

        return cluster_dataset

    def cluster_in_table(
        self,
        cluster_from: Dataset | Table,
        n_clusters: int,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> Table:
        """Cluster the data and store results in a PixelTable table.

        The clustering process applies a clustering algorithm on already vectorized
        representations of the data points and stores results in a PixelTable table specified by `table_path`.
        When the chunk attribute is set, the clustering is performed in chunks to reduce memory consumption.
        If a reducer is provided, the vectors are reduced in dimension before applying the clustering.

        This method is idempotent (i.e., failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will restart the clustering process
        based on the vectors already present in the PixelTable table.

        Args:
            cluster_from: Dataset or Table to select from. If a Dataset is provided, each item should be a
                dictionary with at least the `vectorized_key` and `idx` keys after applying the collate_fn.
                If a Table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_vector` columns.
            n_clusters: Number of clusters to create.
            restrict_to: Optional set of indices to restrict the clustering to.
                If provided, only the selected indices in `restriction_idx_key` will be used for the clustering.
            restriction_idx_key: Column name used to get sample indices for restriction.

        Returns:
            A PixelTable Table containing at least the clustering information in the `cluster_key` column
                and `idx` (index of the sample) and `idx_vector` (index of the vector) columns as well.

        Raises:
            ValueError: If `restriction_idx_key` is not `"idx"` or `"idx_vector"`,
                or if n_clusters is less than or equal to 1.
        """
        if restriction_idx_key not in ("idx", "idx_vector"):
            raise ValueError(
                "restriction_idx_key must be either 'idx' or 'idx_vector'."
            )

        if n_clusters <= 1:
            raise ValueError(
                "cluster_in_table requires n_clusters to be greater than 1."
            )

        logger.info(f"Loading the existing clustering table from {self.table_path}")
        try:
            old_cluster_table = pxt.get_table(
                self.table_path,
                if_not_exists="ignore",
            )
        except Error:
            logger.info(f"No existing clustering table from {self.table_path}")
            old_cluster_table = None

        # clustering table is expected to have a primary key allowing to idempotent updates
        if old_cluster_table is not None:
            check_pxt_table_has_primary_key(old_cluster_table, {"idx_vector"})

        if not self.allow_existing and old_cluster_table is not None:
            raise ValueError(
                f"Table at path {self.table_path} already exists and "
                "allow_existing is set to False."
            )

        if isinstance(cluster_from, Table):
            cluster_table = self._cluster_table_from_table(
                cluster_from=cluster_from,
                old_cluster_table=old_cluster_table,
                restrict_to=restrict_to,
                restriction_idx_key=restriction_idx_key,
            )
        else:
            cluster_table = self._cluster_table_from_dataset(
                cluster_from,
                old_cluster_table,
                restrict_to=restrict_to,
                restriction_idx_key=restriction_idx_key,
            )
            cluster_from = cluster_table

        assert isinstance(cluster_from, Table)

        still_to_cluster_count = self.get_cluster_count(
            cluster_table,
            self.cluster_key,
            restrict_to=restrict_to,
            restriction_idx_key=restriction_idx_key,
        )
        if still_to_cluster_count == 0:
            logger.info(f"Cluster table {self.table_path} already fully clustered")
            return cluster_table
        elif self.distribute:
            self._distributed_cluster(
                cluster_from,
                cluster_table,
                n_clusters,
                restrict_to=restrict_to,
                restriction_idx_key=restriction_idx_key,
            )
        else:
            self._sequential_cluster(
                cluster_from,
                cluster_table,
                n_clusters,
                restrict_to=restrict_to,
                restriction_idx_key=restriction_idx_key,
            )

        logger.info(
            f"Cluster table {self.table_path} successfully clustered in {n_clusters} clusters."
        )

        return cluster_table

    def _cluster_table_from_table(
        self,
        cluster_from: Table,
        old_cluster_table: Table | None,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> Table:
        """Create or validate the cluster table schema from a PixelTable table.

        This private method sets up the table structure with necessary columns for tracking
        clustering status and ensures all rows from the source table are represented.

        Args:
            cluster_from: The source PixelTable table to cluster.
            old_cluster_table: Existing cluster table if resuming, or None.
            restrict_to: Optional set of indices to add to the cluster table.
            restriction_idx_key: Column name used to restrict source rows.

        Returns:
            The cluster table with proper schema and initial rows.

        Raises:
            ValueError: If the source table does not contain the required `vectorized_key` column,
                or if the source table is empty when trying to infer the vectorized data schema.
        """
        minimal_schema = self._MINIMAL_SCHEMA.copy()

        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.label_key is not None:
            minimal_schema[self.label_key] = pxt.String

        cluster_table = get_valid_table(
            table=old_cluster_table
            if old_cluster_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )

        if self.vectorized_key not in cluster_from.columns():
            raise ValueError(
                f"Table at path {self.table_path} does not contain "
                f"the required column {self.vectorized_key}."
            )

        cluster_table_cols = cluster_table.columns()
        if self.cluster_key not in cluster_table_cols:
            cluster_table.add_column(if_exists="error", **{self.cluster_key: pxt.Int})

        if (
            self.include_vectorized_in_table
            and self.vectorized_key not in cluster_table_cols
        ):
            idx_vectors = [
                row["idx_vector"] for row in cluster_from.sample(1).collect()
            ]
            if not idx_vectors:
                raise ValueError(
                    f"Source table at {self.table_path} is empty, cannot sample vectorized data for schema inference."
                )
            sample = get_sample_row_from_idx(
                cluster_from,
                idx_key="idx_vector",
                # make sure to take an existing vector
                idx=idx_vectors[0],
                collate_fn=self.collate_fn,
                expected_keys=[self.vectorized_key],
            )

            vectorized_value = sample[self.vectorized_key].squeeze(0)
            cluster_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized_value.shape, pxt.Float
                    ]
                }
            )

        if cluster_table.count() > 0:
            to_cluster_indices = set(
                [
                    row["idx_vector"]
                    for row in cluster_from.select(cluster_from.idx_vector)
                    .distinct()
                    .collect()
                ]
            )
            already_in_cluster_table = set(
                [
                    row["idx_vector"]
                    for row in cluster_table.select(cluster_table.idx_vector)
                    .distinct()
                    .collect()
                ]
            )
            still_to_cluster = to_cluster_indices.difference(already_in_cluster_table)
            if not still_to_cluster:
                logger.info(
                    f"The cluster table is already initialized in {self.table_path}"
                )
                return cluster_table

        col_list = ["idx", self.vectorized_key]
        if self.to_keep_schema is not None:
            col_list.extend(list(self.to_keep_schema.keys()))

        if self.cluster_key in cluster_from.columns():
            col_list.append(self.cluster_key)

        if self.label_key is not None and self.label_key in cluster_from.columns():
            col_list.append(self.label_key)

        self._add_rows_to_cluster_table_from_dataset(
            cluster_from=GoldPxtTorchDataset(
                cluster_from.select(
                    *[
                        get_expr_from_column_name(cluster_from, col)
                        for col in col_list + ["idx_vector"]
                    ]
                ),
                keep_cache=False,
            ),
            cluster_table=cluster_table,
            include_vectorized=self.include_vectorized_in_table,
            restrict_to=restrict_to,
            restriction_idx_key=restriction_idx_key,
        )

        return cluster_table

    def _cluster_table_from_dataset(
        self,
        cluster_from: Dataset,
        old_cluster_table: Table | None,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> Table:
        """Create or validate the cluster table schema from a PyTorch Dataset.

        This private method sets up the table structure with necessary columns
        and validate that the vectorized column is present in the dataset.

        Args:
            cluster_from: The source PyTorch Dataset to select from.
            old_cluster_table: Existing clustering table if resuming, or None.
            restrict_to: Optional set of indices to add to the cluster table.
            restriction_idx_key: Key used to restrict source samples.

        Returns:
            The clustering table with proper schema.
        """
        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.label_key is not None:
            minimal_schema[self.label_key] = pxt.String

        cluster_table = get_valid_table(
            table=old_cluster_table
            if old_cluster_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )
        cluster_table_cols = cluster_table.columns()
        if self.vectorized_key not in cluster_table_cols:
            sample = get_dataset_sample_dict(
                cluster_from,
                collate_fn=self.collate_fn,
                expected=[self.vectorized_key],
            )

            vectorized_value = (
                sample[self.vectorized_key].squeeze(0).detach().cpu().numpy()
            )
            cluster_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized_value.shape, pxt.Float
                    ]
                }
            )

        if self.cluster_key not in cluster_table_cols:
            cluster_table.add_column(if_exists="error", **{self.cluster_key: pxt.Int})

        self._add_rows_to_cluster_table_from_dataset(
            cluster_from,
            cluster_table,
            include_vectorized=True,
            restrict_to=restrict_to,
            restriction_idx_key=restriction_idx_key,
        )

        return cluster_table

    def _add_rows_to_cluster_table_from_dataset(
        self,
        cluster_from: Dataset,
        cluster_table: Table,
        include_vectorized: bool = False,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> None:
        """Add rows from the source dataset to the cluster table.

        This private method iterates through the dataset in batches and populates the
        cluster table with vectorized data and metadata, skipping already processed samples.

        Args:
            cluster_from: The source PyTorch Dataset.
            cluster_table: The clustering table to populate.
            include_vectorized: Whether to include the vectorized data in the cluster table.
            restrict_to: Optional set of indices to add to the cluster table.
            restriction_idx_key: Key used to restrict source samples.
        """
        dataloader = DataLoader(
            cluster_from,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        already_in_clustering = set(
            [
                row["idx_vector"]
                for row in cluster_table.select(cluster_table.idx_vector).collect()
            ]
        )
        not_empty = (
            len(already_in_clustering) > 0
        )  # allow to filter out already described samples

        ready_to_insert: list[dict[str, Any]] = []

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Initializing rows for the clustering table",
            total=(
                None if hasattr(cluster_from, "__len__") is False else len(dataloader)
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

            if restrict_to is not None:
                to_remove = {
                    int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)
                    for idx in batch[restriction_idx_key]
                } - restrict_to
                if to_remove:
                    batch = filter_batch_from_indices(
                        batch,
                        to_remove,
                        index_key=restriction_idx_key,
                    )
                    if len(batch) == 0:
                        continue

            # Keep only not yet included samples in the batch
            if not_empty:
                batch = filter_batch_from_indices(
                    batch,
                    already_in_clustering,
                    index_key="idx_vector",
                )

                if len(batch) == 0:
                    continue  # all samples already described

            if self.cluster_key not in batch:
                batch[self.cluster_key] = [
                    None for _ in range(len(batch[self.vectorized_key]))
                ]

            already_in_clustering.update(
                [
                    idx.item() if isinstance(idx, torch.Tensor) else idx
                    for idx in batch["idx_vector"]
                ]
            )
            to_insert_keys = ["idx", self.cluster_key]
            if self.to_keep_schema is not None:
                to_insert_keys.extend(list(self.to_keep_schema.keys()))
            if self.label_key is not None and self.label_key in batch:
                to_insert_keys.append(self.label_key)
            if include_vectorized:
                to_insert_keys.append(self.vectorized_key)

            batch_as_list = make_batch_ready_for_table(
                batch,
                to_insert_keys,
                "idx_vector",
            )

            ready_to_insert.extend(batch_as_list)
            if len(ready_to_insert) >= self.min_pxt_insert_size:
                cluster_table.insert(ready_to_insert)
                ready_to_insert = []

        if ready_to_insert:
            cluster_table.insert(ready_to_insert)

    @staticmethod
    def get_cluster_pxt_query(
        table: Table,
        cluster_key: str,
        cluster_idx: int | None = None,
        label_key: str | None = None,
        label_value: str | None = None,
        idx_key: str = "idx_vector",
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> Query:
        """Get the Pixeltable query to access samples clustered in a given cluster.

        Args:
            table: PixelTable table to query.
            cluster_key: Column name used to store the clustering values.
            cluster_idx: Value in the cluster column to filter samples.
            label_key: Optional column name used to filter samples by label.
            label_value: Optional label value to filter samples by label.
            idx_key: Column name used to get sample indices.
            restrict_to: Optional set of indices to restrict the search to.
                If provided, only the selected indices in `restriction_idx_key` will be used for the clustering.
            restriction_idx_key: Column name used to get sample indices for restriction.

        Returns: A Pixeltable Query object that can be used to access
            the samples in the specified cluster (and label if specified).

        Raises:
            ValueError: If only one of `label_key` or `label_value` is provided (both must be set together).
        """
        cluster_col = get_expr_from_column_name(table, cluster_key)
        idx_col = get_expr_from_column_name(table, idx_key)

        if label_value is not None and label_key is not None:
            label_col = get_expr_from_column_name(table, label_key)
            query = (cluster_col == cluster_idx) & (label_col == label_value)
        else:
            if label_key is not None or label_value is not None:
                raise ValueError("label_key and label_value must be set together.")
            query = cluster_col == cluster_idx  # noqa: E712

        if restrict_to is not None:
            restrict_idx_col = get_expr_from_column_name(table, restriction_idx_key)
            query = query & restrict_idx_col.isin(restrict_to)

        return table.where(query).select(idx_col).distinct()

    @staticmethod
    def get_cluster_indices(
        table: Table,
        cluster_key: str = "cluster",
        cluster_idx: int | None = None,
        label_key: str | None = None,
        label_value: str | None = None,
        idx_key: str = "idx_vector",
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> set[int]:
        """Get the indices of samples clustered in a given cluster.

        Args:
            table: PixelTable table to query.
            cluster_key: Column name used to store the clustering values.
            cluster_idx: Value in the cluster column to filter samples.
            label_key: Optional column name used to filter samples by label.
            label_value: Optional label value to filter samples by label.
            idx_key: Column name used to get sample indices.
            restrict_to: Optional set of indices to restrict the search to.
                If provided, only the selected indices in `restriction_idx_key` will be used for the clustering.
            restriction_idx_key: Column name used to get sample indices for restriction.

        Returns: A set of indices corresponding to the samples in the specified cluster (and label if specified).

        Raises:
            ValueError: If only one of `label_key` or `label_value` is provided (both must be set together).
        """
        return set(
            [
                row[idx_key]
                for row in GoldClusterizer.get_cluster_pxt_query(
                    table=table,
                    cluster_key=cluster_key,
                    cluster_idx=cluster_idx,
                    label_key=label_key,
                    label_value=label_value,
                    idx_key=idx_key,
                    restrict_to=restrict_to,
                    restriction_idx_key=restriction_idx_key,
                ).collect()
            ]
        )

    @staticmethod
    def get_cluster_count(
        table: Table,
        cluster_key: str,
        cluster_idx: int | None = None,
        label_key: str | None = None,
        label_value: str | None = None,
        idx_key: str = "idx_vector",
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> int:
        """Get the number of samples clustered in a given cluster.

        Args:
            table: PixelTable table to query.
            cluster_key: Column name used to store the clustering values.
            cluster_idx: Value in the cluster column to filter samples.
            label_key: Optional column name used to filter samples by label.
            label_value: Optional label value to filter samples by label.
            idx_key: Column name used to get sample indices.
            restrict_to: Optional set of indices to restrict the search to.
                If provided, only the selected indices in `restriction_idx_key` will be used for the clustering.
            restriction_idx_key: Column name used to get sample indices for restriction.

        Returns: The number of samples in the specified cluster (and label if specified).

        Raises:
            ValueError: If only one of `label_key` or `label_value` is provided (both must be set together).
        """
        return GoldClusterizer.get_cluster_pxt_query(
            table=table,
            cluster_key=cluster_key,
            cluster_idx=cluster_idx,
            label_key=label_key,
            label_value=label_value,
            idx_key=idx_key,
            restrict_to=restrict_to,
            restriction_idx_key=restriction_idx_key,
        ).count()

    def _sequential_cluster(
        self,
        cluster_from: Table,
        cluster_table: Table,
        n_clusters: int,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> None:
        """Run sequential (single-process) clustering process.

        This private method handles label-stratified clustering if a label_key is configured,
        otherwise performs clustering on the full dataset. It delegates the actual clustering
        to _label_cluster.

        Args:
            cluster_from: The source table with vectorized data.
            cluster_table: The table to store clustering results.
            n_clusters: Number of clusters.

        """
        if self.label_key is not None:
            label_col = get_expr_from_column_name(cluster_table, self.label_key)
            label_values = [
                distinct_item[self.label_key]
                for distinct_item in cluster_table.select(label_col)
                .distinct()
                .collect()
            ]

            for label_idx, label_value in enumerate(label_values):
                label_still_to_cluster_count = GoldClusterizer.get_cluster_count(
                    table=cluster_table,
                    cluster_key=self.cluster_key,
                    label_key=self.label_key,
                    label_value=label_value,
                    restrict_to=restrict_to,
                    restriction_idx_key=restriction_idx_key,
                )
                if label_still_to_cluster_count == 0:
                    logger.info(
                        f"Label '{label_value}' already fully clustered, skipping."
                    )
                    continue

                self._cluster_label(
                    cluster_from,
                    cluster_table,
                    n_clusters,
                    label_value=label_value,
                    restrict_to=restrict_to,
                    restriction_idx_key=restriction_idx_key,
                )

        else:
            still_to_cluster_count = GoldClusterizer.get_cluster_count(
                table=cluster_table,
                cluster_key=self.cluster_key,
                restrict_to=restrict_to,
                restriction_idx_key=restriction_idx_key,
            )
            logger.info(
                f"Clustering {still_to_cluster_count} samples in {n_clusters} clusters."
            )
            self._cluster_label(
                cluster_from,
                cluster_table,
                n_clusters,
                restrict_to=restrict_to,
                restriction_idx_key=restriction_idx_key,
            )

    def _cluster_label(
        self,
        cluster_from: Table,
        cluster_table: Table,
        n_clusters: int,
        label_value: str | None = None,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> None:
        """Perform clustering for a specific label or all data.

        This private method implements chunked clustering.
        It processes data in chunks to manage memory, applies optional dimensionality reduction,
        and updates the cluster table with clustered samples. Chunk assignments for the
        pending label/restriction population are saved before fitting any chunk. A complete
        saved assignment is reused; otherwise the full pending population is reassigned.

        Args:
            cluster_from: The source table with vectorized data.
            cluster_table: The table to store clustering results.
            n_clusters: Number of clusters.
            label_value: Optional label value to filter samples by label.

        Raises:
            ValueError: If an unexpected empty chunk is encountered during clustering.
        """
        cluster_col = get_expr_from_column_name(cluster_table, self.cluster_key)
        vectorized_col = get_expr_from_column_name(cluster_from, self.vectorized_key)
        idx_key = "idx" if self.force_same_cluster else "idx_vector"
        source_idx_col = get_expr_from_column_name(cluster_from, idx_key)
        cluster_idx_col = get_expr_from_column_name(cluster_table, idx_key)

        if label_value is not None:
            assert self.label_key is not None
            label_col = get_expr_from_column_name(cluster_table, self.label_key)
            available_query = (cluster_col == None) & (label_col == label_value)  # noqa: E712 E711
        else:
            available_query = cluster_col == None  # noqa: E711

        if restrict_to is not None:
            restricted_idx_col = get_expr_from_column_name(
                cluster_table, restriction_idx_key
            )
            available_query = available_query & restricted_idx_col.isin(restrict_to)

        to_cluster = cluster_table.where(available_query)
        to_cluster_vector_indices = [
            row["idx_vector"]
            for row in to_cluster.select(cluster_table.idx_vector).collect()
        ]
        available_for_clustering = len(to_cluster_vector_indices)
        if available_for_clustering == 0:
            logger.info(
                f"No samples to cluster for label '{label_value}'."
                if label_value is not None
                else "No samples to cluster."
            )
            return

        # define the chunk assignment in a view of the cluster table
        chunks_view = get_chunk_view(cluster_table)
        chunk_assignment = get_chunk_assignments_from_table(
            chunks_view,
            restrict_to=set(to_cluster_vector_indices),
            max_chunk_size=self.chunk,
            random_state=self.random_state,
        )

        for chunk_indices in chunk_assignment:
            to_cluster_from = cluster_from.where(
                cluster_from.idx_vector.isin(chunk_indices)
            ).select(vectorized_col, source_idx_col)

            if to_cluster_from.count() == 0:
                raise ValueError("Unexpected empty chunk for clustering.")

            to_cluster_for_chunk = [
                (
                    torch.from_numpy(sample[self.vectorized_key]),
                    torch.tensor(sample[idx_key]).unsqueeze(0),
                )
                for sample in to_cluster_from.collect()
            ]
            vectors_list, indices_list = map(list, zip(*to_cluster_for_chunk))
            vectors = torch.stack(vectors_list, dim=0)
            indices = torch.cat(indices_list, dim=0)

            if self.reducer is not None:
                vectors = (
                    self.reducer.fit_transform(vectors)
                    if isinstance(self.reducer, GoldReductionToolWithFit)
                    else self.reducer.transform(vectors)
                )

            cluster_assignment = self.clustering_tool.fit(vectors, n_clusters)

            # update table with cluster information
            for cluster_idx in set(cluster_assignment.tolist()):
                indices_in_cluster = indices[cluster_assignment == cluster_idx].tolist()
                set_value_to_idx_rows(
                    table=cluster_table,
                    col_expr=cluster_col,
                    idx_expr=cluster_idx_col,
                    indices=set(indices_in_cluster),
                    value=cluster_idx,
                )

    def _distributed_cluster(
        self,
        cluster_from: Table,
        cluster_table: Table,
        n_clusters: int,
        restrict_to: set[int] | None = None,
        restriction_idx_key: str = "idx_vector",
    ) -> None:
        """Run distributed clustering process (not implemented).

        Args:
            cluster_from: The source table with vectorized data.
            cluster_table: The table to store the clustering results.
            n_clusters: Number of clusters.

        Raises:
            NotImplementedError: Always raised as distributed mode is not yet implemented.
        """
        raise NotImplementedError("Distributed clustering is not implemented yet.")
