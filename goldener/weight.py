"""Compute and persist per-sample weights from local embedding density.

`GoldDensityWeighingTool` defines the tensor contract, while
`GoldDensityWeigher` stages vectors in PixelTable and stores resumable scores.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable

import numpy as np
import pixeltable as pxt
import torch
from pixeltable import Error
from pixeltable.catalog import Table
from torch.utils.data import DataLoader, Dataset

from goldener.clusterize import get_random_chunk_assignment
from goldener.do import GoldDoerWithTable
from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    check_pxt_table_has_primary_key,
    get_array_column_shapes_from_query,
    get_expr_from_column_name,
    get_valid_table,
    make_batch_ready_for_table,
)
from goldener.reduce import GoldReductionTool, GoldReductionToolWithFit
from goldener.select import DistanceType
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

    Args:
        device: Device used for tensor computation. Defaults to ``"cpu"``.

    The input matrix has shape `(num_vectors, feature_dim)` and implementations
    return a one-dimensional vector with one score per input row.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate the finite floating matrix accepted by density tools."""
        validate_vectorized_input(x, "GoldDensityWeighingTool")

    @abstractmethod
    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        """Return one finite nonnegative density score per input row."""


class GoldKnnDensityWeighingTool(GoldDensityWeighingTool):
    """Estimate density from mean distance to the nearest neighbors.

    Each score is the mean k-neighbor distance raised to `power`, in the
    distance function's original units. `eps` is an absolute floor on the
    mean distance; it gives duplicates a small positive score.
    Inputs must contain at least `k + 2` vectors.

    Attributes:
        k: Positive number of neighbors per row when computing scores. The search retrieves
            `k + 1` entries and discards the query row itself. Inputs require at
            least `k + 2` vectors. Defaults to `5`.
        power: Finite nonnegative exponent applied to the mean neighbor
            distance. Defaults to `0.5`.
        distance_func: Pairwise distance callable selected by `distance`.
            Defaults to the Euclidean distance function.
        distance_batch_size: Positive number of query rows compared per batch
            against the full input population. Defaults to `1024`.
        eps: Finite positive floor on the mean neighbor distance before
            exponentiation. Defaults to `1e-12`.
        device: Torch device used for computation. Defaults to CPU; returned
            scores retain the input tensor's device.
    """

    def __init__(
        self,
        k: int = 5,
        power: float = 0.5,
        distance: DistanceType = DistanceType.EUCLIDEAN,
        distance_batch_size: int = 1024,
        eps: float = 1e-12,
        device: torch.device | str = "cpu",
    ) -> None:
        """Configure deterministic k-nearest-neighbor density scores.

        Args:
            k: Number of neighbors used to compute each row's score. The search retrieves
                `k + 1` entries and discards the query row itself. Inputs require
                at least `k + 2` vectors. Defaults to `5`.
            power: Exponent applied to the mean neighbor distance in original
                metric units. Defaults to `0.5`.
            distance: Pairwise distance function. Defaults to Euclidean.
            distance_batch_size: Number of query rows compared per distance
                batch. The full input remains the reference population.
            eps: Absolute lower bound for the mean neighbor distance before
                exponentiation.
            device: Computation device. Defaults to CPU.
        """
        super().__init__(device)
        validate_positive_integer(k, "k")
        validate_positive_integer(distance_batch_size, "distance_batch_size")
        validate_positive_number(power, "power", allow_zero=True)
        validate_positive_number(eps, "eps")
        self.k = k
        self.power = float(power)
        self.distance_func = distance.distance_function
        self.distance_batch_size = distance_batch_size
        self.eps = float(eps)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate vectors and require at least `k + 2` input rows."""
        super()._validate_input(x)
        if len(x) < self.k + 2:
            raise ValueError(
                f"x must contain at least k + 2 ({self.k + 2}) vectors; got {len(x)}."
            )

    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw kNN distance scores, preserving input dtype and device.

        Distances are computed in `torch.float64` on the configured device.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)`,
                with at least `k + 2` rows.

        Returns:
            A nonnegative tensor shaped `(num_vectors,)` on the input device and dtype.

        Raises:
            ValueError: If the input is invalid or has fewer than `k + 2` rows.
        """
        self._validate_input(x)
        input_device, input_dtype = x.device, x.dtype
        x = x.to(device=self.device, dtype=torch.float64)

        n = len(x)
        means = []

        # Batch queries against the full population so batching does not change neighbors.
        for start in range(0, n, self.distance_batch_size):
            query = x[start : start + self.distance_batch_size]

            # compute the distance between the batch and the full set
            distances = self.distance_func(query, x)  # distance_batch_size, n tensor

            # make the distance with itself the smallest possible
            local_rows = torch.arange(len(query), device=x.device)
            global_rows = torch.arange(start, start + len(query), device=x.device)
            distances[local_rows, global_rows] = -torch.inf

            # find the average of the k closest vectors, itself excluded
            neighbor_distances = distances.topk(self.k + 1, largest=False, dim=1).values
            means.append(neighbor_distances[:, 1:].mean(dim=1))

        # Floor distances before exponentiation to give duplicates positive scores.
        mean_dist = torch.cat(means).clamp_min(self.eps)

        scores = mean_dist.pow(self.power)
        return scores.to(device=input_device, dtype=input_dtype)


class GoldDensityWeigher(GoldDoerWithTable):
    """Persist raw per-vector scores derived from local embedding density.

    Dataset inputs provide one finite floating vector per item under
    `vectorized_key`. Missing `idx_vector` values are generated from item
    positions, and missing `idx` values default to those vector IDs. Table
    inputs must contain the vector column and both IDs. `idx_vector` is unique;
    multiple vectors may share one sample `idx`.

    Scores are stored by vector in `weight_key`. Inputs and scoring settings
    must remain unchanged when resuming a result table. `max_batches` limits
    the source prefix processed by each call. `chunk` groups newly staged
    vectors for independent scoring; each chunk must contain at least `k + 2`
    vectors for the default kNN tool. `random_state` seeds each new assignment.
    A fit-capable reducer is fitted separately for each chunk; a transform-only
    reducer is applied without fitting. Completed scores are reused on later
    calls.

    `weigh_in_dataset` returns a snapshot containing IDs, scores, available
    vectors, and configured retained columns. `drop_table=True` drops the
    persisted table after that snapshot is created.

    Attributes:
        table_path: PixelTable path used to persist vector rows and scores.
        weigher: Tensor tool that computes one raw nonnegative score per vector;
            defaults to a new `GoldKnnDensityWeighingTool` with `k=5`.
        reducer: Optional reducer applied to each pending chunk before scoring;
            defaults to `None`.
        chunk: Maximum vectors per independently scored chunk; defaults to
            `None`, grouping all vectors newly ingested during a call.
        collate_fn: Optional dataset collator; defaults to the sequence-preserving
            collator.
        vectorized_key: Source vector column/key. Defaults to `vectorized`.
        weight_key: Column storing raw per-vector scores. Defaults to `weight`.
        to_keep_schema: Optional additional source columns retained in the table;
            defaults to `None`.
        min_pxt_insert_size: Minimum staged rows accumulated before insertion;
            defaults to `100`.
        batch_size: Number of dataset items loaded per batch; defaults to `1`.
        num_workers: DataLoader worker count; defaults to `0`.
        allow_existing: Whether an existing result table may be resumed;
            defaults to `True`.
        distribute: Distributed processing capability; currently unsupported
            and defaults to `False`.
        drop_table: Whether to drop the table after creating a dataset snapshot;
            defaults to `False`.
        max_batches: Maximum source batches processed on each call; defaults to
            `None` (all batches).
        random_state: Optional seed used for each new chunk assignment; defaults
            to `None`.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
        "idx_vector": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        table_path: str,
        weigher: GoldDensityWeighingTool | None = None,
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
    ) -> None:
        """Configure persisted density weighting.

        Args:
            table_path: PixelTable result path.
            weigher: Tensor tool returning one finite, nonnegative score per input
                vector. Defaults to a new kNN density weigher with `k=5`.
            reducer: Optional reducer applied to each new density chunk. Reducers
                supporting fitting are fitted independently on each chunk;
                transform-only reducers are applied without fitting.
            chunk: Maximum vectors per independently scored chunk. Defaults to
                `None`, placing all vectors ingested by one call in one chunk.
                Each chunk must contain at least `k + 2` vectors for the default
                kNN weigher (7 vectors with its default `k=5`).
            collate_fn: Dataset collator; defaults to
                `collate_keeping_sequences_as_sequences`.
            vectorized_key: Input vector column/key. Defaults to `vectorized`.
            weight_key: Raw vector score column. Defaults to `weight`.
            to_keep_schema: Additional source columns to retain in the result
                table. Defaults to `None`.
            min_pxt_insert_size: Minimum staged rows accumulated before insertion;
                defaults to `100`.
            batch_size: Dataset ingestion batch size; defaults to `1`.
            num_workers: DataLoader worker count; defaults to `0`.
            allow_existing: Whether an existing result table may be resumed;
                defaults to `True`.
            distribute: Distributed processing is unsupported and defaults to
                `False`.
            drop_table: Drop the result table after creating a dataset snapshot;
                defaults to `False`.
            max_batches: Maximum source prefix batches to ingest on each call;
                defaults to `None` (all batches).
            random_state: Optional seed reused for each new chunk assignment;
                defaults to `None`.

        Raises:
            NotImplementedError: If distributed processing is requested.
            ValueError: If `chunk` is not a positive integer or `None`.
        """
        if distribute:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldDensityWeigher."
            )
        validate_positive_integer(chunk, "chunk", allow_none=True)
        super().__init__(
            allow_existing=allow_existing,
            drop_table=drop_table,
            max_batches=max_batches,
            collate_fn=collate_fn,
            to_keep_schema=to_keep_schema,
            min_pxt_insert_size=min_pxt_insert_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.table_path = table_path
        self.weigher = weigher if weigher is not None else GoldKnnDensityWeighingTool()
        self.reducer = reducer
        self.chunk = chunk
        self.vectorized_key = vectorized_key
        self.weight_key = weight_key
        self.random_state = random_state
        self._distribute = False

    @property
    def distribute(self) -> bool:
        """Whether unsupported distributed processing is enabled."""
        return self._distribute

    @distribute.setter
    def distribute(self, value: bool) -> None:
        """Reject enabling unsupported distributed processing.

        Args:
            value: Must be `False`.

        Raises:
            NotImplementedError: If `value` is `True`.
        """
        if value:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldDensityWeigher."
            )
        self._distribute = False

    def weigh_in_table(self, to_weigh: Dataset | Table) -> Table:
        """Stage source vectors and persist raw per-vector density scores.

        `max_batches` limits the source prefix processed on this call, and that
        prefix is scored immediately. Later calls can add new rows without
        changing existing raw scores. Dataset items must provide a finite
        floating 1D tensor under `vectorized_key`; absent IDs are generated.
        Table inputs require `vectorized_key`, `idx`, and unique `idx_vector`
        columns. Multiple vector rows may refer to one sample `idx`.

        Args:
            to_weigh: Dataset or PixelTable table containing vector rows.

        Returns:
            The persistent result table, with `idx`, `idx_vector`, `density_chunk`,
            and `weight_key` columns, the vector column when available, and
            configured retained columns.

        Raises:
            ValueError: If required columns, vectors, IDs, or stored schema are
                invalid, or if a resumed table is incompatible.
        """
        weighing_table = self.setup_weighing_table(to_weigh)
        dataset = self._source_dataset(to_weigh)
        self._add_rows_to_weighing_table_from_dataset(dataset, weighing_table)
        self._assign_new_chunks(weighing_table)
        self._sequential_weigh(weighing_table)
        return weighing_table

    def weigh_in_dataset(self, to_weigh: Dataset | Table) -> GoldPxtTorchDataset:
        """Return a dataset snapshot for the currently processed source prefix.

        The snapshot contains per-vector raw scores. Use `get_weights_per_idx`
        to average available vector scores by sample, and call
        `normalize_weights` explicitly if mean-one sample weights are wanted.
        With `drop_table=True`, the persistent result table is dropped after the
        snapshot has been created.

        Args:
            to_weigh: Dataset or PixelTable table containing vector rows.

        Returns:
            A dataset snapshot with IDs, scores, available vectors, and retained
            source columns.

        Raises:
            ValueError: If any currently staged vector has no score or input
                validation fails.
        """
        table = self.weigh_in_table(to_weigh)
        if self._count_nulls(table, self.weight_key) > 0:
            raise ValueError(
                "Density scores are incomplete; resume scoring before creating a dataset."
            )
        keys = ["idx", "idx_vector", self.weight_key]
        if self.vectorized_key in table.columns():
            keys.append(self.vectorized_key)
        keys.extend(
            key for key in (self.to_keep_schema or {}) if key in table.columns()
        )
        view = table.select(
            *[get_expr_from_column_name(table, key) for key in dict.fromkeys(keys)]
        )
        dataset = GoldPxtTorchDataset(view)
        if self.drop_table:
            pxt.drop_table(self.table_path)
        return dataset

    def setup_weighing_table(self, to_weigh: Dataset | Table) -> Table:
        """Get or create the persisted table and its scoring columns.

        Args:
            to_weigh: Source used to infer the vector shape when needed.

        Returns:
            The result table keyed by unique `idx_vector`.

        Raises:
            ValueError: If an existing table is disallowed, has an incompatible
                primary key, or existing rows lack their vector column.
        """
        try:
            old_table = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            old_table = None
        if old_table is not None:
            check_pxt_table_has_primary_key(old_table, {"idx_vector"})
        if not self.allow_existing and old_table is not None:
            raise ValueError(
                f"Table at path {self.table_path} already exists and allow_existing is False."
            )

        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema:
            minimal_schema.update(self.to_keep_schema)
        table = get_valid_table(
            old_table if old_table is not None else self.table_path,
            minimal_schema,
            primary_key="idx_vector",
        )
        if self.vectorized_key not in table.columns():
            vector_shape = self._get_vector_shape(to_weigh)
            if vector_shape is not None:
                table.add_column(
                    **{self.vectorized_key: pxt.Array[vector_shape, pxt.Float]}  # type: ignore[misc]
                )
            elif table.count() > 0:
                raise ValueError(
                    "Existing density rows are missing their vector column."
                )
        for key, column_type in (
            ("density_chunk", pxt.Int),
            (self.weight_key, pxt.Float),
        ):
            if key not in table.columns():
                table.add_column(if_exists="error", **{key: column_type})
        return table

    def _get_vector_shape(self, source: Dataset | Table) -> tuple[int, ...] | None:
        if isinstance(source, Table):
            if self.vectorized_key not in source.columns():
                raise ValueError(
                    f"Source table does not contain vector column '{self.vectorized_key}'."
                )
            shapes = get_array_column_shapes_from_query(
                source.select(get_expr_from_column_name(source, self.vectorized_key))
            )
            return shapes.get(self.vectorized_key)
        try:
            sample = get_dataset_sample_dict(
                source,
                collate_fn=self.collate_fn,
                expected=[self.vectorized_key],
            )
        except (IndexError, StopIteration):
            return None
        vector = sample[self.vectorized_key]
        if (
            isinstance(vector, torch.Tensor)
            and vector.ndim == 2
            and vector.shape[0] == 1
        ):
            vector = vector[0]
        return tuple(vector.shape)

    def _source_dataset(self, source: Dataset | Table) -> Dataset:
        if isinstance(source, Table):
            required = [self.vectorized_key, "idx", "idx_vector"]
            required.extend(self.to_keep_schema or {})
            missing = set(required).difference(source.columns())
            if missing:
                raise ValueError(
                    f"Source table is missing required columns: {missing}."
                )
            return GoldPxtTorchDataset(
                source.select(
                    *[
                        get_expr_from_column_name(source, key)
                        for key in dict.fromkeys(required)
                    ]
                )
            )
        return source

    def _add_rows_to_weighing_table_from_dataset(
        self, source: Dataset, table: Table
    ) -> None:
        """Insert the configured source prefix, skipping stored vector IDs."""
        loader = DataLoader(
            source,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        existing = {
            int(row["idx_vector"]) for row in table.select(table.idx_vector).collect()
        }
        ready: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for batch_idx, batch in enumerate(loader):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            if self.vectorized_key not in batch:
                raise ValueError(f"Dataset batch is missing '{self.vectorized_key}'.")
            count = len(batch[self.vectorized_key])
            if "idx_vector" not in batch:
                start = batch_idx * self.batch_size
                batch["idx_vector"] = list(range(start, start + count))
            if "idx" not in batch:
                batch["idx"] = batch["idx_vector"]
            vector_ids = [
                int(value.item()) if isinstance(value, torch.Tensor) else int(value)
                for value in batch["idx_vector"]
            ]
            if len(set(vector_ids)) != len(vector_ids) or seen_ids.intersection(
                vector_ids
            ):
                raise ValueError("Dataset batch contains duplicate idx_vector values.")
            seen_ids.update(vector_ids)
            if existing:
                batch = filter_batch_from_indices(
                    batch, existing, index_key="idx_vector"
                )
                if not batch:
                    continue
            for vector in batch[self.vectorized_key]:
                if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
                    raise ValueError("Each vector must be a one-dimensional tensor.")
                if not vector.is_floating_point() or not torch.isfinite(vector).all():
                    raise ValueError(
                        "Each vector must contain finite floating point values."
                    )
            batch[self.vectorized_key] = [
                vector.detach().cpu().numpy() for vector in batch[self.vectorized_key]
            ]
            for key in self.to_keep_schema or {}:
                if key in batch:
                    batch[key] = [
                        value.detach().cpu().numpy()
                        if isinstance(value, torch.Tensor) and value.ndim > 0
                        else value
                        for value in batch[key]
                    ]
            insert_keys = [self.vectorized_key, "idx"] + list(self.to_keep_schema or {})
            rows = make_batch_ready_for_table(
                batch, insert_keys, index_key="idx_vector"
            )
            ready.extend(rows)
            existing.update(
                int(value.item()) if isinstance(value, torch.Tensor) else int(value)
                for value in batch["idx_vector"]
            )
            if len(ready) >= self.min_pxt_insert_size:
                table.insert(ready)
                ready = []
        if ready:
            table.insert(ready)

    def _assign_new_chunks(self, table: Table) -> None:
        """Assign only unassigned rows to chunk IDs after existing groups."""
        unassigned = (
            table.where(table.density_chunk == None)  # noqa: E711
            .select(  # noqa: E711
                table.idx_vector
            )
            .collect()
        )
        ids = sorted(int(row["idx_vector"]) for row in unassigned)
        if not ids:
            return
        assigned_ids = [
            int(row["density_chunk"])
            for row in table.select(table.density_chunk).distinct().collect()
            if row["density_chunk"] is not None
        ]
        first_chunk = max(assigned_ids, default=-1) + 1
        chunks = get_random_chunk_assignment(ids, self.chunk, self.random_state)
        table.batch_update(
            [
                {"idx_vector": idx_vector, "density_chunk": first_chunk + chunk_idx}
                for chunk_idx, chunk in enumerate(chunks)
                for idx_vector in chunk
            ]
        )

    def _sequential_weigh(self, table: Table) -> None:
        """Score pending groups and persist raw scores by vector ID."""
        pending_chunks = (
            table.where(
                get_expr_from_column_name(table, self.weight_key) == None  # noqa: E711
            )
            .select(table.density_chunk)
            .distinct()
            .collect()
        )
        chunk_ids = sorted(int(row["density_chunk"]) for row in pending_chunks)
        for chunk_id in chunk_ids:
            rows = (
                table.where(table.density_chunk == chunk_id)
                .select(
                    table.idx_vector, table[self.vectorized_key], table[self.weight_key]
                )
                .collect()
            )
            completed = [row[self.weight_key] is not None for row in rows]
            if all(completed):
                continue
            if any(completed):
                raise ValueError("Stored raw scores are incomplete within a chunk.")
            vectors = torch.stack(
                [torch.as_tensor(row[self.vectorized_key]) for row in rows]
            )
            scores = self._compute_weights(vectors)
            table.batch_update(
                [
                    {
                        "idx_vector": int(row["idx_vector"]),
                        self.weight_key: float(score.item()),
                    }
                    for row, score in zip(rows, scores)
                ]
            )

    def _compute_weights(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the optional reducer and validate one score per input vector."""
        matrix = vectors
        if self.reducer is not None:
            matrix = (
                self.reducer.fit_transform(matrix)
                if isinstance(self.reducer, GoldReductionToolWithFit)
                else self.reducer.transform(matrix)
            )
            if (
                not isinstance(matrix, torch.Tensor)
                or matrix.ndim != 2
                or len(matrix) != len(vectors)
            ):
                raise ValueError("Reducer must preserve the number of vector rows.")
        scores = self.weigher.weigh(matrix)
        if (
            not isinstance(scores, torch.Tensor)
            or scores.shape != (len(matrix),)
            or not torch.isfinite(scores).all()
            or (scores < 0).any()
        ):
            raise ValueError(
                "Density weigher must return one finite nonnegative score per vector."
            )
        return scores

    @staticmethod
    def _count_nulls(table: Table, key: str) -> int:
        return table.where(get_expr_from_column_name(table, key) == None).count()  # noqa: E711

    @staticmethod
    def get_weights_per_idx(
        table: Table, weight_key: str = "weight", idx_key: str = "idx"
    ) -> dict[int, float]:
        """Return mean available raw vector score per source sample `idx`.

        Unscored rows are skipped, so this mapping can describe a successfully
        completed prefix while a later chunk is still pending.

        Args:
            table: Result table containing sample IDs and scores.
            weight_key: Score column name. Defaults to `weight`.
            idx_key: Sample ID column name. Defaults to `idx`.

        Returns:
            Mapping from sample IDs to the arithmetic mean of their available
            raw per-vector scores. Samples without scored vectors are omitted.
        """
        rows = table.select(table[idx_key], table[weight_key]).collect()
        scores_by_idx: dict[int, list[float]] = {}
        for row in rows:
            if row[weight_key] is None:
                continue
            scores_by_idx.setdefault(int(row[idx_key]), []).append(
                float(row[weight_key])
            )
        means = {}
        for idx, scores in scores_by_idx.items():
            maximum = max(scores)
            means[idx] = (
                0.0
                if maximum == 0
                else maximum * (sum(score / maximum for score in scores) / len(scores))
            )
        return means

    @staticmethod
    def normalize_weights(weights: dict[int, float]) -> dict[int, float]:
        """Return mean-one scores for supplied sample scores without writing.

        Empty input returns an empty mapping. All-zero input returns ones. Values
        must be finite and nonnegative. Uncomputed rows are not included; callers
        can pass the partial mapping from `get_weights_per_idx` directly.

        Args:
            weights: Mapping from sample IDs to finite nonnegative scores.

        Returns:
            A new mapping scaled to mean one, or the empty mapping for empty input.

        Raises:
            ValueError: If any score is boolean, negative, or nonfinite.
        """
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value < 0
            for value in weights.values()
        ):
            raise ValueError("Weights must be finite and nonnegative.")
        if not weights:
            return {}
        maximum = max(weights.values())
        if maximum == 0:
            return {idx: 1.0 for idx in weights}
        scaled = {idx: value / maximum for idx, value in weights.items()}
        mean = sum(scaled.values()) / len(scaled)
        return {idx: value / mean for idx, value in scaled.items()}

    @staticmethod
    def get_weights_per_idx_vector(
        table: Table, weight_key: str = "weight"
    ) -> dict[int, float]:
        """Return available raw scores keyed by individual `idx_vector`.

        Rows without a score are omitted.

        Args:
            table: Result table containing vector IDs and scores.
            weight_key: Score column name. Defaults to `weight`.

        Returns:
            Mapping from vector IDs to their raw per-vector scores.
        """
        rows = table.select(table.idx_vector, table[weight_key]).collect()
        return {
            int(row["idx_vector"]): float(row[weight_key])
            for row in rows
            if row[weight_key] is not None
        }

    def estimate_computation_time(self, to_weigh: Dataset | Table) -> float | None:
        """Time only the kNN scoring of a minimal read-only Table sample.

        This estimate is the measured duration of scoring exactly `k + 2`
        vectors; it is not an estimate of total pipeline time. Dataset sources,
        custom weighers, configured reducers, and nonempty tables smaller than
        `k + 2` return `None`. On the supported default-tool, no-reducer path,
        empty tables return `0.0`. Sampling is read-only and does not consume
        random state.

        Args:
            to_weigh: Dataset or PixelTable table; datasets are unsupported and
                return `None` without sampling.

        Returns:
            Measured seconds for the scoring sample, `0.0` for an empty table on
            the supported default-tool path, or `None` when unsupported.
        """
        if (
            isinstance(to_weigh, Dataset)
            or type(self.weigher) is not GoldKnnDensityWeighingTool
            or self.reducer is not None
        ):
            return None
        minimum_size = self.weigher.k + 2
        rows = (
            to_weigh.select(to_weigh[self.vectorized_key]).limit(minimum_size).collect()
        )
        if not rows:
            return 0.0
        if len(rows) < minimum_size:
            return None
        vectors = torch.stack(
            [torch.as_tensor(row[self.vectorized_key]).reshape(-1) for row in rows]
        )
        import time

        start = time.perf_counter()
        self._compute_weights(vectors)
        return time.perf_counter() - start
