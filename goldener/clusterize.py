from abc import ABC, abstractmethod
from logging import getLogger
from typing import Callable, Any

import numpy as np
import pixeltable as pxt
from pixeltable import Error
from pixeltable.catalog import Table

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from goldener.pxt_utils import (
    set_value_to_idx_rows,
    GoldPxtTorchDataset,
    get_expr_from_column_name,
    get_valid_table,
    make_batch_ready_for_table,
    check_pxt_table_has_primary_key,
)
from goldener.reduce import GoldReducer
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import (
    filter_batch_from_indices,
)


logger = getLogger(__name__)


class GoldClusteringTool(ABC):
    """Run clustering on input vectors."""

    @abstractmethod
    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """Fit the clustering tool from a given set of vectors into n clusters.

        Args:
            x: Input vectors to select from.
            n_clusters: Number of clusters to form.

        Returns: The cluster assignments for each input vector as a 1D tensor of cluster indices.
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Assign the input vectors to the clusters.

        Args:
            x: Input vectors to transform.

        Returns: The cluster assignments for each input vector as a 1D tensor of cluster indices.
        """
        pass


class GoldRandomClusteringTool(GoldClusteringTool):
    """Chunk data randomly into clusters of almost equal size."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """Randomly assign each input vector to clusters of roughly the same size.

        Args:
            x: Input vectors to select from.

        Returns: The cluster assignments for each input vector as a 1D tensor of cluster indices.
        """
        total = len(x)

        if n_clusters > total:
            raise ValueError(
                f"Number of clusters ({n_clusters}) cannot be greater than "
                f"the number of samples ({total})."
            )

        cluster_assignement = [i % n_clusters for i in range(total)]
        np.random.default_rng(self.random_state).shuffle(cluster_assignement)

        return torch.tensor(
            cluster_assignement,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Random clustering tool does not support predict.")


class GoldClusterizer:
    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
        "idx_vector": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        table_path: str,
        clustering_tool: GoldClusteringTool,
        reducer: GoldReducer | None = None,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        vectorized_key: str = "vectorized",
        cluster_key: str = "cluster",
        class_key: str | None = None,
        to_keep_schema: dict[str, type] | None = None,
        min_pxt_insert_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 0,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.table_path = table_path
        self.clustering_tool = clustering_tool
        self.reducer = reducer
        if chunk is not None and chunk <= 0:
            raise ValueError("chunk must be a positive integer or None.")
        self.chunk = chunk
        self.collate_fn = collate_fn
        self.vectorized_key = vectorized_key
        self.cluster_key = cluster_key
        self.class_key = class_key
        self.to_keep_schema = to_keep_schema
        self.min_pxt_insert_size = min_pxt_insert_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_existing = allow_existing
        self.distribute = distribute
        self.drop_table = drop_table
        self.max_batches = max_batches
        self.random_state = random_state

    def cluster_in_dataset(
        self, cluster_from: Dataset | Table, n_clusters: int
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
                If the collate_fn is None, the dataset is expected to directly provide such batches.
                If a Table is provided, it should contain at least the `vectorized_key` and `idx` columns.
            n_clusters: Number of clusters to create.
        Returns:
            A GoldPxtTorchDataset containing at least the clustering information in the `cluster_key` key
                and `idx` (index of the sample) and `idx_vector` (index of the vector) keys as well.
        """

        cluster_table = self.cluster_in_table(cluster_from, n_clusters)

        cluster_dataset = GoldPxtTorchDataset(cluster_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(cluster_table)

        return cluster_dataset

    def cluster_in_table(self, cluster_from: Dataset | Table, n_clusters: int) -> Table:
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
                If the collate_fn is None, the dataset is expected to directly provide such batches.
                If a Table is provided, it should contain at least the `vectorized_key`, `idx` and `idx_vector` columns.
            n_clusters: Number of clusters to create.

        Returns:
            A PixelTable Table containing at least the clustering information in the `cluster_key` column
                and `idx` (index of the sample) and `idx_vector` (index of the vector) columns as well.

        Raises:
            ValueError: If n_clusters is less than or equal to 1.
        """
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

        # clustering table are expected to have a primary key allowing to idempotent updates
        if old_cluster_table is not None:
            check_pxt_table_has_primary_key(old_cluster_table, set(["idx_vector"]))

        if not self.allow_existing and old_cluster_table is not None:
            raise ValueError(
                f"Table at path {self.table_path} already exists and "
                "allow_existing is set to False."
            )

        if isinstance(cluster_from, Table):
            cluster_table = self._cluster_table_from_table(
                cluster_from=cluster_from,
                old_cluster_table=old_cluster_table,
            )
        else:
            cluster_table = self._cluster_table_from_dataset(
                cluster_from, old_cluster_table
            )
            cluster_from = cluster_table

        assert isinstance(cluster_from, Table)

        # define the number of element to sample
        total_size = cluster_from.select(cluster_from.idx).distinct().count()

        if (
            len(self.get_cluster_sample_indices(cluster_table, self.cluster_key))
            == total_size
        ):
            logger.info(f"Cluster table {self.table_path} already fully clustered")
            return cluster_table
        elif self.distribute:
            self._distributed_cluster(cluster_from, cluster_table, n_clusters)
        else:
            self._sequential_cluster(cluster_from, cluster_table, n_clusters)

        logger.info(
            f"Cluster table {self.table_path} successfully clustered in {n_clusters} clusters."
        )

        return cluster_table

    def _cluster_table_from_table(
        self, cluster_from: Table, old_cluster_table: Table | None
    ) -> Table:
        """Create or validate the cluster table schema from a PixelTable table.

        This private method sets up the table structure with necessary columns for tracking
        clustering status and ensures all rows from the source table are represented.

        Args:
            cluster_from: The source PixelTable table to cluster.
            old_cluster_table: Existing cluster table if resuming, or None.

        Returns:
            The cluster table with proper schema and initial rows.
        """
        minimal_schema = self._MINIMAL_SCHEMA

        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.class_key is not None:
            minimal_schema[self.class_key] = pxt.String

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

        if self.cluster_key not in cluster_table.columns():
            cluster_table.add_column(if_exists="error", **{self.cluster_key: pxt.Int})

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

        if self.class_key is not None and self.class_key in cluster_from.columns():
            col_list.append(self.class_key)

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
            include_vectorized=False,
        )

        return cluster_table

    def _cluster_table_from_dataset(
        self, cluster_from: Dataset, old_cluster_table: Table | None
    ) -> Table:
        """Create or validate the cluster table schema from a PyTorch Dataset.

        This private method sets up the table structure with necessary columns
        and validate that the vectorized column is present in the dataset.

        Args:
            cluster_from: The source PyTorch Dataset to select from.
            old_cluster_table: Existing clustering table if resuming, or None.

        Returns:
            The clustering table with proper schema.
        """
        minimal_schema = self._MINIMAL_SCHEMA
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.class_key is not None:
            minimal_schema[self.class_key] = pxt.String

        cluster_table = get_valid_table(
            table=old_cluster_table
            if old_cluster_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )

        if self.vectorized_key not in cluster_table.columns():
            sample = get_dataset_sample_dict(
                cluster_from,
                collate_fn=self.collate_fn,
                expected=[self.vectorized_key],
            )

            vectorized_value = sample[self.vectorized_key].detach().cpu().numpy()
            cluster_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized_value.shape, pxt.Float
                    ]
                }
            )

        if self.cluster_key not in cluster_table.columns():
            cluster_table.add_column(if_exists="error", **{self.cluster_key: pxt.Int})

        self._add_rows_to_cluster_table_from_dataset(
            cluster_from, cluster_table, include_vectorized=True
        )

        return cluster_table

    def _add_rows_to_cluster_table_from_dataset(
        self,
        cluster_from: Dataset,
        cluster_table: Table,
        include_vectorized: bool = False,
    ) -> None:
        """Add rows from the source dataset to the cluster table.

        This private method iterates through the dataset in batches and populates the
        cluster table with vectorized data and metadata, skipping already processed samples.

        Args:
            cluster_from: The source PyTorch Dataset.
            cluster_table: The clustering table to populate.
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
            if self.class_key is not None and self.class_key in batch:
                to_insert_keys.append(self.class_key)
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
    def get_cluster_sample_indices(
        table: Table,
        cluster_key: str,
        class_key: str | None = None,
        class_value: str | None = None,
        idx_key: str = "idx",
    ) -> set[int]:
        """Get the indices of samples clustered in a given cluster.

        Args:
            table: PixelTable table to query.
            cluster_idx: Value in the cluster column to filter clustered samples.
            cluster_key: Column name used to store the clustering values.
            class_key: Optional column name used to filter samples by class.
            class_value: Optional class value to filter samples by class.
            idx_key: Column name used to get sample indices.
        """
        idx_col = get_expr_from_column_name(table, idx_key)
        cluster_col = get_expr_from_column_name(table, cluster_key)
        if class_value is not None and class_key is not None:
            class_col = get_expr_from_column_name(table, class_key)
            query = (cluster_col != None) & (class_col == class_value)  # noqa: E711
        else:
            if class_key is not None or class_value is not None:
                raise ValueError("class_key and class_value must be set together.")
            query = cluster_col != None  # noqa: E711

        return set(
            [
                row[idx_key]
                for row in table.where(query)  # noqa: E712
                .select(idx_col)
                .distinct()
                .collect()
            ]
        )

    def _sequential_cluster(
        self,
        cluster_from: Table,
        cluster_table: Table,
        n_clusters: int,
    ) -> None:
        """Run sequential (single-process) clustering process.

        This private method handles class-stratified clustering if a class_key is configured,
        otherwise performs clustering on the full dataset. It delegates the actual clustering
        to _class_cluster.

        Args:
            cluster_from: The source table with vectorized data.
            cluster_table: The table to store clustering results.
            n_clusters: Number of clusters.
        """
        if self.class_key is not None:
            class_col = get_expr_from_column_name(cluster_table, self.class_key)
            class_values = [
                distinct_item[self.class_key]
                for distinct_item in cluster_table.select(class_col)
                .distinct()
                .collect()
            ]

            for class_idx, class_value in enumerate(class_values):
                already_clustered = len(
                    self.get_cluster_sample_indices(
                        table=cluster_table,
                        cluster_key=self.cluster_key,
                        class_key=self.class_key,
                        class_value=class_value,
                    )
                )
                class_still_to_cluster_count = (
                    cluster_table.where(
                        class_col == class_value  # noqa: E712
                    )
                    .select(cluster_table.idx)
                    .distinct()
                    .count()
                )

                class_still_to_cluster_count = (
                    class_still_to_cluster_count - already_clustered
                )
                if class_still_to_cluster_count == 0:
                    logger.info(
                        f"Class '{class_value}' already fully clustered, skipping."
                    )
                    continue
                elif class_still_to_cluster_count < 0:
                    raise ValueError(
                        "The size of the cluster table has decreased since the 1st clustering computation"
                    )

                self._cluster_class(
                    cluster_from,
                    cluster_table,
                    n_clusters,
                    class_value=class_value,
                )

        else:
            already_clustered = len(
                self.get_cluster_sample_indices(
                    table=cluster_table,
                    cluster_key=self.cluster_key,
                )
            )
            sample_count = cluster_table.select(cluster_table.idx).distinct().count()
            still_to_cluster_count = sample_count - already_clustered
            logger.info(
                f"Clustering {still_to_cluster_count} samples in {n_clusters} clusters."
            )
            self._cluster_class(
                cluster_from,
                cluster_table,
                n_clusters,
            )

    def _cluster_class(
        self,
        cluster_from: Table,
        cluster_table: Table,
        n_clusters: int,
        class_value: str | None = None,
    ) -> None:
        """Perform clustering for a specific class or all data.

        This private method implements chunked clustering.
        It processes data in chunks to manage memory, applies optional dimensionality reduction,
        and updates the cluster table with clustered samples.

        Args:
            cluster_from: The source table with vectorized data.
            cluster_table: The table to store clustering results.
            n_clusters: Number of clusters.
            class_value: Optional class value to filter samples by class.
        """
        cluster_col = get_expr_from_column_name(cluster_table, self.cluster_key)
        vectorized_col = get_expr_from_column_name(cluster_from, self.vectorized_key)

        if class_value is not None:
            assert self.class_key is not None
            class_col = get_expr_from_column_name(cluster_table, self.class_key)
            available_query = (cluster_col == None) & (class_col == class_value)  # noqa: E712 E711
        else:
            available_query = cluster_col == None  # noqa: E711

        if self.chunk is None:
            chunk_assignment = [
                [
                    row["idx_vector"]
                    for row in cluster_table.where(available_query)
                    .select(cluster_table.idx_vector)
                    .collect()
                ]
            ]
        else:
            to_cluster = cluster_table.where(available_query)
            to_cluster_vector_indices = [
                row["idx_vector"]
                for row in to_cluster.select(cluster_table.idx_vector).collect()
            ]
            available_for_clustering = len(to_cluster_vector_indices)
            chunk_count = available_for_clustering // self.chunk
            random_assignment = (
                GoldRandomClusteringTool(random_state=self.random_state)
                .fit(
                    torch.randn(
                        available_for_clustering, 1
                    ),  # dummy input for random clustering
                    chunk_count,
                )
                .tolist()
            )
            chunk_assignment = [
                [
                    vector_idx
                    for vect_pos, vector_idx in enumerate(to_cluster_vector_indices)
                    if random_assignment[vect_pos] == chunk_idx
                ]
                for chunk_idx in range(chunk_count)
            ]

        for chunk_indices in chunk_assignment:
            to_cluster_from = cluster_from.where(
                cluster_from.idx.isin(chunk_indices)
            ).select(vectorized_col, cluster_from.idx_vector)

            to_cluster_for_chunk = [
                (
                    torch.from_numpy(sample[self.vectorized_key]),
                    torch.tensor(sample["idx_vector"]).unsqueeze(0),
                )
                for sample in to_cluster_from.collect()
            ]
            vectors_list, indices_list = map(list, zip(*to_cluster_for_chunk))
            vectors = torch.stack(vectors_list, dim=0)
            indices = torch.cat(indices_list, dim=0)

            if self.reducer is not None:
                vectors = self.reducer.fit_transform(vectors)

            cluster_assignment = self.clustering_tool.fit(vectors, n_clusters)

            # update table with selected indices
            for cluster_idx in set(cluster_assignment):
                indices_in_cluster = indices[cluster_assignment == cluster_idx].tolist()
                set_value_to_idx_rows(
                    table=cluster_table,
                    col_expr=cluster_col,
                    idx_expr=cluster_table.idx_vector,
                    indices=set(indices_in_cluster),
                    value=cluster_idx,
                )

    def _distributed_cluster(
        self,
        cluster_from: Table,
        cluster_table: Table,
        n_clusters: int,
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
