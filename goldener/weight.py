"""Compute and persist per-sample weights from local embedding density.

`GoldDensityWeighingTool` defines the tensor contract, while
`GoldDensityWeigher` stages vectors in PixelTable and stores resumable scores.
"""

from abc import ABC, abstractmethod
from logging import getLogger
import json
import secrets
from numbers import Integral
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
    get_array_column_shapes_from_table,
    get_valid_table,
)
from goldener.reduce import GoldReductionTool, GoldReductionToolWithFit
from goldener.select import DistanceType

logger = getLogger(__name__)


def _as_index(value: Any, name: str) -> int:
    """Convert an integer index without truncating booleans or floats."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool or value.is_floating_point():
            raise ValueError(f"{name} must be an integer scalar.")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer scalar.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return int(value)


def _column_classname(table: Table, name: str) -> str:
    """Return a PixelTable column's serialized base type name."""
    return table[name].col_type.as_dict()["_classname"]


class GoldDensityWeighingTool(ABC):
    """Compute one positive novelty weight for each vector in a 2D tensor.

    Args:
        device: Device used for tensor computation. Defaults to ``"cpu"``.

    The input matrix has shape `(num_vectors, feature_dim)` and implementations
    return a one-dimensional vector with one score per input row.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate the finite floating matrix accepted by density tools."""
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("x must be a 2D tensor with positive feature width.")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("x must contain only finite floating point values.")

    @abstractmethod
    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        """Return a finite positive novelty weight for every input row."""


class GoldKnnDensityWeighingTool(GoldDensityWeighingTool):
    """Estimate novelty from mean distance to the nearest neighbors.

    Scores are normalized to mean one. They are an inverse crowding proxy,
    not calibrated density probabilities. Query batching bounds temporary
    distance memory without changing the reference population.
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
        """Configure deterministic k-nearest-neighbor novelty scores.

        Args:
            k: Number of neighbors used for each row, capped at `n - 1`.
            power: Exponent applied to the mean neighbor distance. Defaults to
                `0.5` to damp large sparse-region scores.
            distance: Pairwise distance function. Defaults to Euclidean.
            distance_batch_size: Number of query rows compared per distance
                batch. The full input remains the reference population.
            eps: Relative lower bound for zero/tiny mean neighbor distances.
            device: Computation device. Defaults to CPU.
        """
        super().__init__(device)
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if (
            isinstance(distance_batch_size, bool)
            or not isinstance(distance_batch_size, int)
            or distance_batch_size <= 0
        ):
            raise ValueError("distance_batch_size must be a positive integer.")
        if isinstance(power, bool) or not np.isfinite(power) or power < 0:
            raise ValueError("power must be finite and nonnegative.")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.k = k
        self.power = float(power)
        self.distance_func = distance.distance_function
        self.distance_batch_size = distance_batch_size
        self.eps = float(eps)

    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean-one kNN distance scores, preserving input dtype/device.

        Args:
            x: Finite floating tensor shaped `(num_vectors, feature_dim)`.

        Returns:
            A positive tensor shaped `(num_vectors,)` on the input device and dtype.
        """
        self._validate_input(x)
        n = len(x)
        if n == 0:
            return x.new_empty((0,))
        if n == 1 or self.power == 0:
            return x.new_ones((n,))

        work = x.to(self.device)
        dtype = (
            torch.float32
            if work.dtype in (torch.float16, torch.bfloat16)
            else work.dtype
        )
        work = work.to(dtype)
        if not torch.isfinite(work).all():
            raise ValueError(
                "x cannot be represented finitely on the configured device."
            )
        if self.distance_func is torch.cdist:
            scale = work.abs().max()
            if scale > 0:
                work = work / scale

        means = []
        k = min(self.k, n - 1)
        for start in range(0, n, self.distance_batch_size):
            query = work[start : start + self.distance_batch_size]
            distances = self.distance_func(query, work)
            local_rows = torch.arange(len(query), device=work.device)
            global_rows = torch.arange(start, start + len(query), device=work.device)
            distances[local_rows, global_rows] = torch.inf
            means.append(distances.topk(k, largest=False, dim=1).values.mean(dim=1))
        mean_dist = torch.cat(means)
        positive = mean_dist[mean_dist > 0]
        if positive.numel() == 0:
            return x.new_ones((n,))
        # Normalize before exponentiation to avoid overflow; eps is relative to
        # the nonzero distance scale, so duplicate vectors stay well-defined.
        scale = positive.mean()
        log_scores = self.power * (mean_dist / scale).clamp_min(self.eps).log()
        scores = torch.exp(log_scores - log_scores.max())
        scores = scores.clamp_min(torch.finfo(scores.dtype).tiny)
        scores = scores / scores.mean()
        scores = scores.to(device=x.device, dtype=x.dtype)
        scores = scores.clamp_min(torch.finfo(x.dtype).tiny)
        scores = scores / scores.mean()
        if not torch.isfinite(scores).all() or (scores <= 0).any():
            raise ValueError(
                "Computed density weights are not representable in the input dtype."
            )
        return scores


class GoldDensityWeigher(GoldDoerWithTable):
    """Stage vectors and persist resumable density-derived sample weights.

    Custom tools and reducers are identified by component type when resuming;
    callers must keep their configuration and behavior unchanged across runs.
    Incomplete ingestion remains in the result table with null weights, and
    `weigh_in_dataset` raises until all vectors have been ingested and scored.
    """

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
        """Initialize persisted density weighting.

        Args:
            table_path: Local PixelTable path for vector scores and sample weights.
            weigher: Tensor tool; defaults to a fresh `GoldKnnDensityWeighingTool`.
            reducer: Optional reducer applied to vectors before weighing.
            chunk: Optional positive chunk size. Density is computed independently
                within each randomly assigned chunk; `None` uses the full population.
            collate_fn: Dataset collator, defaulting to the shared sequence-preserving collator.
            vectorized_key: Source vector column, default `vectorized`.
            weight_key: Final per-sample weight column, default `weight`.
            to_keep_schema: Additional source columns to retain.
            min_pxt_insert_size: Minimum pending rows before insertion.
            batch_size: Dataset ingestion batch size.
            num_workers: DataLoader worker count.
            allow_existing: Whether an existing output table may be resumed.
            distribute: Distributed processing is unsupported.
            drop_table: Drop output and companion state tables after snapshotting a dataset.
            max_batches: Maximum new dataset batches ingested per call.
            random_state: Seed for a persisted chunk assignment; `None` stores a generated seed.
        """
        if distribute:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldDensityWeigher."
            )
        reserved = {
            "idx",
            "idx_vector",
            "density_raw",
            "density_chunk",
            "density_finalized",
        }
        if (
            weight_key in reserved
            or vectorized_key in reserved
            or weight_key == vectorized_key
            or (
                to_keep_schema
                and reserved.union({weight_key, vectorized_key}).intersection(
                    to_keep_schema
                )
            )
        ):
            raise ValueError(
                "vectorized_key, weight_key, and kept columns cannot use reserved internal names."
            )
        super().__init__(
            allow_existing,
            drop_table,
            max_batches,
            collate_fn,
            to_keep_schema,
            min_pxt_insert_size,
            batch_size,
            num_workers,
        )
        self.table_path = table_path
        self.state_path = f"{table_path}_density_state"
        self.weigher = weigher if weigher is not None else GoldKnnDensityWeighingTool()
        self.reducer = reducer
        self.chunk = chunk
        self.vectorized_key = vectorized_key
        self.weight_key = weight_key
        self.random_state = random_state
        self._distribute = False

    @property
    def chunk(self) -> int | None:
        """Maximum number of vectors per randomly assigned density population."""
        return self._chunk

    @chunk.setter
    def chunk(self, value: int | None) -> None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("chunk must be a positive integer or None.")
        self._chunk = value

    def _configuration(self) -> str:
        tool_config = None
        if type(self.weigher) is GoldKnnDensityWeighingTool:
            tool_config = {
                "k": self.weigher.k,
                "power": self.weigher.power,
                "distance": self.weigher.distance_func.__name__,
                "distance_batch_size": self.weigher.distance_batch_size,
                "eps": self.weigher.eps,
                "device": str(self.weigher.device),
            }
        return json.dumps(
            {
                "vectorized_key": self.vectorized_key,
                "weight_key": self.weight_key,
                "chunk": self.chunk,
                "random_state": self.random_state,
                "batch_size": self.batch_size,
                "collate_type": f"{type(self.collate_fn).__module__}.{type(self.collate_fn).__qualname__}",
                "weigher_type": f"{type(self.weigher).__module__}.{type(self.weigher).__qualname__}",
                "weigher": tool_config,
                "reducer_type": None
                if self.reducer is None
                else f"{type(self.reducer).__module__}.{type(self.reducer).__qualname__}",
                "kept_schema": {
                    key: str(value)
                    for key, value in (self.to_keep_schema or {}).items()
                },
            },
            sort_keys=True,
            default=str,
        )

    @property
    def distribute(self) -> bool:
        """Whether unsupported distributed processing is enabled."""
        return self._distribute

    @distribute.setter
    def distribute(self, value: bool) -> None:
        """Reject enabling unsupported distributed processing."""
        if value:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldDensityWeigher."
            )
        self._distribute = False

    def _tables(self, source: Dataset | Table) -> tuple[Table, Table]:
        try:
            result = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            result = None
        try:
            state = pxt.get_table(self.state_path, if_not_exists="ignore")
        except Error:
            state = None
        if (result is not None or state is not None) and not self.allow_existing:
            raise ValueError(
                f"Table at path {self.table_path} already exists and allow_existing is False."
            )
        if (
            result is not None
            and result.count() > 0
            and (state is None or state.count() == 0)
        ):
            raise ValueError(
                "Existing density rows have no companion state; refusing to adopt an unrelated table."
            )
        if result is None and state is not None and state.count() > 0:
            raise ValueError(
                "Existing density state has no result table; refusing an unsafe resume."
            )
        if result is not None:
            check_pxt_table_has_primary_key(result, {"idx_vector"})
            required_types = {
                "idx": "IntType",
                "idx_vector": "IntType",
            }
            optional_types = {
                self.vectorized_key: "ArrayType",
                "density_raw": "FloatType",
                self.weight_key: "FloatType",
                "density_chunk": "IntType",
                "density_finalized": "BoolType",
            }
            if result.count() > 0:
                missing = set(optional_types).difference(result.columns())
                if missing:
                    raise ValueError(
                        f"Existing density result table is missing required columns: {missing}."
                    )
            required_types.update(
                {
                    name: expected
                    for name, expected in optional_types.items()
                    if name in result.columns()
                }
            )
            for name, expected in required_types.items():
                if (
                    name in result.columns()
                    and _column_classname(result, name) != expected
                ):
                    raise ValueError(
                        f"Existing density result column '{name}' must have type {expected}."
                    )
        if state is not None:
            missing_state = {"key", "value"}.difference(state.columns())
            if missing_state:
                raise ValueError(
                    f"Existing density state table is missing required columns: {missing_state}."
                )
            if (
                _column_classname(state, "key") != "StringType"
                or _column_classname(state, "value") != "StringType"
            ):
                raise ValueError(
                    "Existing density state columns must both have string types."
                )
            check_pxt_table_has_primary_key(state, {"key"})
        schema: dict[str, type] = {
            "idx": pxt.Required[pxt.Int],
            "idx_vector": pxt.Required[pxt.Int],
        }
        if self.to_keep_schema:
            schema.update(self.to_keep_schema)
        result = get_valid_table(
            result if result is not None else self.table_path, schema, "idx_vector"
        )
        if self.vectorized_key not in result.columns():
            try:
                sample = self._sample(source)
                vector = sample[self.vectorized_key]
                if vector.ndim == 2 and vector.shape[0] == 1:
                    vector = vector[0]
            except (IndexError, StopIteration):
                vector = torch.empty((0,), dtype=torch.float32)
            if isinstance(vector, torch.Tensor):
                vector = vector.detach().cpu().numpy()
            result.add_column(
                **{self.vectorized_key: pxt.Array[np.asarray(vector).shape, pxt.Float]}  # type: ignore[misc]
            )
        for key, typ in (
            ("density_raw", pxt.Float),
            (self.weight_key, pxt.Float),
            ("density_chunk", pxt.Int),
            ("density_finalized", pxt.Bool),
        ):
            if key not in result.columns():
                result.add_column(**{key: typ})  # type: ignore[arg-type]
        state_schema = {"key": pxt.Required[pxt.String], "value": pxt.String}
        state = get_valid_table(
            state if state is not None else self.state_path, state_schema, "key"
        )
        return result, state

    def _sample(self, source: Dataset | Table) -> dict[str, Any]:
        if isinstance(source, Table):
            rows = source.select(source[self.vectorized_key]).limit(1).collect()
            if not rows:
                shapes = get_array_column_shapes_from_table(source)
                if self.vectorized_key not in shapes:
                    raise ValueError(
                        "An empty source table must declare an array shape for its vector column."
                    )
                return {self.vectorized_key: torch.empty(shapes[self.vectorized_key])}
            return {self.vectorized_key: torch.as_tensor(rows[0][self.vectorized_key])}
        return self.collate_fn([source[0]])

    def weigh_in_table(self, to_weigh: Dataset | Table) -> Table:
        """Stage vectors, compute local-density scores, and return the result table.

        Each result row represents an `idx_vector`, keeps its source sample
        `idx`, and receives the final sample weight. Incomplete dataset ingestion
        returns staged rows with nullable `weight` values so a later call can
        resume. Internal `density_raw`, `density_chunk`, and
        `density_finalized` columns are reserved for this doer.
        """
        return self._run(to_weigh)

    def weigh_in_dataset(self, to_weigh: Dataset | Table) -> GoldPxtTorchDataset:
        """Return a snapshot-backed dataset after all weights are finalized.

        Raises:
            ValueError: If the configured ingestion limit leaves the result
                incomplete. The staged table remains available for resuming.
        """
        table = self.weigh_in_table(to_weigh)
        if self._state_value(table, "finalized") != "1":
            raise ValueError(
                "Density weights are incomplete; resume ingestion and scoring before creating a dataset."
            )
        public_keys = [
            "idx",
            "idx_vector",
            self.vectorized_key,
            self.weight_key,
        ] + list(self.to_keep_schema or {})
        view = table.select(*[table[key] for key in dict.fromkeys(public_keys)])
        dataset = GoldPxtTorchDataset(view)
        if self.drop_table:
            pxt.drop_table(self.table_path, force=True)
            pxt.drop_table(self.state_path, force=True)
        return dataset

    def _state_value(self, table: Table, key: str) -> str | None:
        state = pxt.get_table(self.state_path)
        assert state is not None
        rows = state.where(state.key == key).select(state.value).collect()
        return rows[0]["value"] if rows else None

    def _set_state(self, state: Table, key: str, value: str) -> None:
        if state.where(state.key == key).count():
            state.where(state.key == key).update({"value": value})
        else:
            state.insert([{"key": key, "value": value}])

    def _run(self, source: Dataset | Table) -> Table:
        table, state = self._tables(source)
        config = self._configuration()
        old_config = self._state_value(table, "config")
        if old_config is not None and old_config != config:
            raise ValueError(
                "Density result table configuration differs from the stored run; use a new table_path."
            )
        self._set_state(state, "config", config)
        self._validate_population(source, table)
        if self._state_value(table, "complete") != "1":
            self._ingest(source, table, state)
        if self._state_value(table, "complete") == "1":
            self._score(table, state)
            self._finalize(table, state)
        return table

    def _validate_population(self, source: Dataset | Table, table: Table) -> None:
        scoring_started = self._state_value(table, "scoring_started") == "1"
        if table.count() == 0 or (
            self._state_value(table, "complete") == "1" and not scoring_started
        ):
            return
        staged = {
            int(row["idx_vector"]) for row in table.select(table.idx_vector).collect()
        }
        current: set[int]
        if isinstance(source, Table):
            mapping: dict[int, int] = {}
            for row in source.select(source.idx, source.idx_vector).collect():
                idx_vector = _as_index(row["idx_vector"], "idx_vector")
                idx = _as_index(row["idx"], "idx")
                if idx_vector in mapping:
                    raise ValueError("Source contains duplicate idx_vector values.")
                mapping[idx_vector] = idx
            current = set(mapping)
        else:
            current = set()
            mapping = {}
            cursor = 0
            loader = DataLoader(
                source,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for batch in loader:
                size = len(batch[self.vectorized_key])
                values = batch.get("idx_vector", list(range(cursor, cursor + size)))
                indices = batch.get("idx", values)
                for i, value in enumerate(values):
                    idx_vector = _as_index(value, "idx_vector")
                    idx = _as_index(indices[i], "idx")
                    if idx_vector in mapping:
                        raise ValueError(
                            "Dataset contains duplicate idx_vector values."
                        )
                    mapping[idx_vector] = idx
                    current.add(idx_vector)
                cursor += size
        if not staged.issubset(current):
            raise ValueError(
                "Source vector population removed staged rows after density scoring began."
            )
        staged_mapping = {
            int(row["idx_vector"]): int(row["idx"])
            for row in table.select(table.idx_vector, table.idx).collect()
        }
        if any(
            mapping.get(idx_vector) != idx for idx_vector, idx in staged_mapping.items()
        ):
            raise ValueError(
                "Source idx-to-idx_vector mapping changed after density scoring began."
            )
        if (
            scoring_started
            and self._state_value(table, "complete") == "1"
            and current != staged
        ):
            raise ValueError(
                "Source vector population changed after density scoring began."
            )

    def _ingest(self, source: Dataset | Table, table: Table, state: Table) -> None:
        if isinstance(source, Table):
            existing = {
                row["idx_vector"] for row in table.select(table.idx_vector).collect()
            }
            source_ids = set()
            table_pending = []
            selected = [self.vectorized_key, "idx", "idx_vector"] + list(
                self.to_keep_schema or {}
            )
            for row in source.select(
                *[source[key] for key in dict.fromkeys(selected)]
            ).collect():
                iv = _as_index(row["idx_vector"], "idx_vector")
                _as_index(row["idx"], "idx")
                if iv in source_ids:
                    raise ValueError("Source contains duplicate idx_vector values.")
                source_ids.add(iv)
                if iv not in existing:
                    vector = np.asarray(row[self.vectorized_key])
                    if (
                        vector.ndim != 1
                        or not np.issubdtype(vector.dtype, np.floating)
                        or not np.isfinite(vector).all()
                    ):
                        raise ValueError(
                            "Each source vector must be a finite floating one-dimensional array."
                        )
                    table_pending.append(
                        {key: row[key] for key in dict.fromkeys(selected)}
                    )
            if table_pending:
                table.batch_update(table_pending, if_not_exists="insert")
            self._set_state(state, "complete", "1")
            return
        existing = {
            int(row["idx_vector"]) for row in table.select(table.idx_vector).collect()
        }
        loader = DataLoader(
            source,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        dataset_pending: list[dict[str, Any]] = []
        processed = 0
        exhausted = True
        cursor = 0
        seen_ids: set[int] = set()
        for batch in loader:
            vectors = batch[self.vectorized_key]
            start = cursor
            cursor += len(vectors)
            vector_ids = batch.get("idx_vector", list(range(start, cursor)))
            indices = batch.get("idx", vector_ids)
            batch_ids = [_as_index(value, "idx_vector") for value in vector_ids]
            if seen_ids.intersection(batch_ids) or len(set(batch_ids)) != len(
                batch_ids
            ):
                raise ValueError("Dataset contains duplicate idx_vector values.")
            seen_ids.update(batch_ids)
            is_new_batch = any(idx_vector not in existing for idx_vector in batch_ids)
            if (
                is_new_batch
                and self.max_batches is not None
                and processed >= self.max_batches
            ):
                exhausted = False
                break
            if is_new_batch:
                processed += 1
            for i, vector in enumerate(vectors):
                idx = _as_index(indices[i], "idx")
                iv = _as_index(vector_ids[i], "idx_vector")
                if iv in existing:
                    continue
                if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
                    raise ValueError("Each dataset vector must be a 1D tensor.")
                if not vector.is_floating_point() or not torch.isfinite(vector).all():
                    raise ValueError(
                        "Each dataset vector must contain finite floating point values."
                    )
                row = {
                    "idx": idx,
                    "idx_vector": iv,
                    self.vectorized_key: vector.detach().cpu().numpy()
                    if isinstance(vector, torch.Tensor)
                    else vector,
                }
                for key in self.to_keep_schema or {}:
                    value = batch[key][i]
                    row[key] = (
                        value.detach().cpu().numpy()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                dataset_pending.append(row)
                existing.add(iv)
            if len(dataset_pending) >= self.min_pxt_insert_size:
                table.batch_update(dataset_pending, if_not_exists="insert")
                dataset_pending = []
        if dataset_pending:
            table.batch_update(dataset_pending, if_not_exists="insert")
        if exhausted:
            self._set_state(state, "complete", "1")

    def _score(self, table: Table, state: Table) -> None:
        rows = table.select(
            table.idx_vector,
            table[self.vectorized_key],
            table.density_raw,
            table.density_chunk,
        ).collect()
        if not rows:
            return
        result_rows: list[dict[str, Any]] = list(rows)
        order = sorted(
            range(len(result_rows)), key=lambda i: int(result_rows[i]["idx_vector"])
        )
        result_rows = [result_rows[i] for i in order]
        ids = [int(row["idx_vector"]) for row in result_rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Result table contains duplicate idx_vector values.")
        vectors = [torch.as_tensor(row[self.vectorized_key]) for row in result_rows]
        if any(v.ndim != 1 for v in vectors):
            raise ValueError("Each staged vector must be one-dimensional.")
        widths = {v.shape[0] for v in vectors}
        if len(widths) != 1:
            raise ValueError(
                "Vectors in the staged population have conflicting dimensions."
            )
        if self.chunk is None:
            assignments = [0] * len(rows)
        else:
            seed = self._state_value(table, "seed")
            if seed is None:
                seed = str(
                    self.random_state
                    if self.random_state is not None
                    else secrets.randbelow(2**31 - 1)
                )
                self._set_state(state, "seed", seed)
            assignments = [-1] * len(rows)
            chunks = get_random_chunk_assignment(ids, self.chunk, int(seed))
            for chunk_idx, chunk_ids in enumerate(chunks):
                for idx in chunk_ids:
                    assignments[ids.index(idx)] = chunk_idx
        stored_assignments = [row["density_chunk"] for row in result_rows]
        if any(value is not None for value in stored_assignments):
            if any(value is None for value in stored_assignments):
                raise ValueError(
                    "Stored density chunk assignments are incomplete; refusing an unsafe resume."
                )
            assignments = [int(value) for value in stored_assignments]
        else:
            table.batch_update(
                [
                    {"idx_vector": idx, "density_chunk": assignment}
                    for idx, assignment in zip(ids, assignments)
                ]
            )
        self._set_state(state, "scoring_started", "1")
        by_chunk: dict[int, list[int]] = {}
        for i, assignment in enumerate(assignments):
            by_chunk.setdefault(assignment, []).append(i)
        for chunk_id, positions in by_chunk.items():
            todo = [i for i in positions if result_rows[i]["density_raw"] is None]
            if not todo:
                continue
            matrix = torch.stack([vectors[i] for i in positions])
            scores = self._compute_weights(matrix)
            if (
                scores.shape != (len(positions),)
                or not torch.isfinite(scores).all()
                or (scores <= 0).any()
            ):
                raise ValueError(
                    "Density weigher must return one finite positive score per vector."
                )
            todo_positions = set(todo)
            updates = [
                {"idx_vector": ids[row_idx], "density_raw": float(scores[j].item())}
                for j, row_idx in enumerate(positions)
                if row_idx in todo_positions
            ]
            table.batch_update(updates, if_not_exists="ignore")

    def _compute_weights(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply optional reduction and validate scores for one fixed population."""
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
            or (scores <= 0).any()
        ):
            raise ValueError(
                "Density weigher must return one finite positive score per vector."
            )
        return scores

    def _finalize(self, table: Table, state: Table) -> None:
        rows = table.select(table.idx, table.idx_vector, table.density_raw).collect()
        if not rows:
            self._set_state(state, "finalized", "1")
            return
        if any(row["density_raw"] is None for row in rows):
            return
        grouped: dict[int, list[tuple[int, float]]] = {}
        for row in rows:
            grouped.setdefault(int(row["idx"]), []).append(
                (int(row["idx_vector"]), float(row["density_raw"]))
            )
        max_score = max(score for vals in grouped.values() for _, score in vals)
        means = {
            idx: float(np.mean([score / max_score for _, score in vals]))
            for idx, vals in grouped.items()
        }
        norm = float(np.mean(list(means.values())))
        means = {idx: score / norm for idx, score in means.items()}
        if any(not np.isfinite(score) or score <= 0 for score in means.values()):
            raise ValueError("Final sample weights are not finite and positive.")
        updates = [
            {
                "idx_vector": iv,
                self.weight_key: means[idx],
                "density_finalized": True,
            }
            for idx, vals in grouped.items()
            for iv, _ in vals
        ]
        table.batch_update(updates, if_not_exists="ignore")
        self._set_state(state, "finalized", "1")

    @staticmethod
    def get_weights_per_idx(
        table: Table, weight_key: str = "weight", idx_key: str = "idx"
    ) -> dict[int, float]:
        """Return finalized sample weights keyed by source sample `idx`."""
        rows = table.select(table[idx_key], table[weight_key]).collect()
        weights: dict[int, float] = {}
        for row in rows:
            if row[weight_key] is None:
                raise ValueError(
                    "Density weights are incomplete; finish ingestion and scoring first."
                )
            weights.setdefault(int(row[idx_key]), float(row[weight_key]))
        return weights

    @staticmethod
    def get_weights_per_idx_vector(
        table: Table, weight_key: str = "weight"
    ) -> dict[int, float]:
        """Return sample-normalized weights keyed by individual vector `idx_vector`."""
        rows = table.select(table.idx_vector, table[weight_key]).collect()
        if any(row[weight_key] is None for row in rows):
            raise ValueError(
                "Density weights are incomplete; finish ingestion and scoring first."
            )
        return {int(row["idx_vector"]): float(row[weight_key]) for row in rows}

    def estimate_computation_time(self, to_weigh: Dataset | Table) -> float | None:
        """Time a small deterministic tensor sample without writing PixelTable state.

        Custom tools, reducers, and Dataset inputs return ``None`` because their
        full random state and data-loading side effects cannot be snapshotted
        safely. Table inputs use a read-only sample.
        """
        if (
            type(self.weigher) is not GoldKnnDensityWeighingTool
            or self.reducer is not None
        ):
            logger.info(
                "Cannot estimate density weighting time for custom or reduced computation."
            )
            return None
        if isinstance(to_weigh, Dataset):
            return None
        if isinstance(to_weigh, Table):
            rows = to_weigh.select(to_weigh[self.vectorized_key]).limit(2).collect()
            vectors = [torch.as_tensor(row[self.vectorized_key]) for row in rows]
        if not vectors:
            return 0.0
        matrix = torch.stack([vector.reshape(-1) for vector in vectors])
        import time

        start = time.perf_counter()
        self._compute_weights(matrix)
        return time.perf_counter() - start
