"""Score vectorized samples by distance from a fitted reference bank.

`GoldNoveltyScoringTool` defines the tensor interface and
`GoldNearestNeighborNoveltyScoringTool` implements exact nearest-reference distance.
`GoldNoveltyScorer` fits on reference vectors and persists scores for query vectors.

Scores remain vector-indexed, so samples with multiple vectors can use the maximum
vector novelty as their sample-level result.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable

import numpy as np
import pixeltable as pxt
import torch
from pixeltable import Error
from pixeltable.catalog import Table
from pixeltable.type_system import ColumnType
from torch.utils.data import DataLoader, Dataset

from goldener.do import GoldDoerWithTable
from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    check_pxt_table_has_primary_key,
    get_expr_from_column_name,
    get_valid_table,
)
from goldener.reduce import GoldReductionToolWithFit
from goldener.select import DistanceType

logger = getLogger(__name__)


class GoldNoveltyScoringTool(ABC):
    """Score vectors against a fitted reference using a distance measure.

    Inputs have shape `(num_vectors, feature_dim)` and scores have shape
    `(num_vectors,)`. Higher scores indicate greater novelty. Fitted state is kept
    in memory.

    Attributes:
        distance: Distance metric used by the tool.
        device: Device used for distance computation.
        fitted: Read-only indicator for whether a reference bank is available.
    """

    def __init__(
        self,
        distance: DistanceType = DistanceType.COSINE,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the abstract novelty scoring contract.

        Args:
            distance: Distance selector used by implementations. Defaults to cosine.
            device: Device used for computation. Defaults to CPU.
        """
        self.distance = distance
        self.device = torch.device(device)

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Whether a reference has been fitted.

        Returns:
            True when this tool can score against a reference bank.
        """

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            shape = getattr(x, "shape", None)
            raise ValueError(
                "GoldNoveltyScoringTool only accepts 2D tensors "
                f"(num_vectors, feature_dim). Got shape {shape}."
            )
        if x.shape[1] <= 0:
            raise ValueError("feature_dim must be greater than zero.")
        if not x.is_floating_point():
            raise ValueError("Novelty vectors must have a floating-point dtype.")
        if not torch.isfinite(x).all():
            raise ValueError("Novelty vectors must contain only finite values.")

    @abstractmethod
    def fit(self, x: torch.Tensor) -> None:
        """Fit the reference bank from 2D vectors.

        Args:
            x: Nonempty floating reference vectors of shape `(num_vectors, feature_dim)`.

        Raises:
            ValueError: If vectors are empty, non-finite, non-floating, or not 2D.
        """

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return one novelty score per input row.

        Args:
            x: Floating query vectors of shape `(num_vectors, feature_dim)`.

        Returns:
            A 1D tensor on the input device and with the input dtype.

        Raises:
            ValueError: If the tool is unfitted or inputs do not satisfy the tensor contract.
        """


class GoldNearestNeighborNoveltyScoringTool(GoldNoveltyScoringTool):
    """Use each query's minimum distance to any fitted reference vector.

    Exact computation stores the complete reference bank in memory. Pairwise
    temporary allocations are bounded by `distance_batch_size` query and reference
    rows. Cosine distance uses Goldener's existing helper, including distance 1 for
    a zero vector against any vector. Scores preserve query device and dtype.

    Attributes:
        distance_batch_size: Maximum rows in each query/reference computation block.
        reference_bank: Cloned fitted vectors on the configured device, or None.
        fitted: Whether a reference bank has been fitted.
    """

    def __init__(
        self,
        distance: DistanceType = DistanceType.COSINE,
        device: torch.device | str = "cpu",
        distance_batch_size: int = 1024,
    ) -> None:
        """Create a blocked exact nearest-reference scorer.

        Args:
            distance: Pairwise distance selector. Defaults to cosine.
            device: Computation device. Defaults to CPU.
            distance_batch_size: Positive query and reference block size. Defaults to 1024.

        Raises:
            ValueError: If `distance_batch_size` is not a positive integer.
        """
        super().__init__(distance=distance, device=device)
        if (
            not isinstance(distance_batch_size, int)
            or isinstance(distance_batch_size, bool)
            or distance_batch_size <= 0
        ):
            raise ValueError("distance_batch_size must be a positive integer.")
        self.distance_batch_size = distance_batch_size
        self.reference_bank: torch.Tensor | None = None

    @property
    def fitted(self) -> bool:
        """Whether a reference bank is available."""
        return self.reference_bank is not None

    def fit(self, x: torch.Tensor) -> None:
        """Replace the fitted reference bank with an independent clone of `x`.

        Args:
            x: Nonempty finite floating reference matrix `(num_vectors, feature_dim)`.

        Raises:
            ValueError: If `x` is empty or violates the 2D floating tensor contract.
        """
        self._validate_input(x)
        if x.shape[0] == 0:
            raise ValueError("The novelty reference must not be empty.")
        self.reference_bank = x.detach().to(self.device).clone()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return exact minimum reference distance for each query row.

        Args:
            x: Query vectors of shape `(num_vectors, feature_dim)`.

        Returns:
            A `(num_vectors,)` tensor on the query's device and with its dtype; higher
            scores mean more novel.

        Raises:
            ValueError: If the tool is unfitted, dimensions mismatch, or a result is non-finite.
        """
        if self.fitted:
            raise ValueError("Fit the novelty scorer before calling score.")
        assert self.reference_bank is not None
        self._validate_input(x)

        if x.shape[1] != self.reference_bank.shape[1]:
            raise ValueError(
                "Query feature_dim must match the fitted reference feature_dim."
            )
        if x.shape[0] == 0:
            return torch.empty((0,), dtype=x.dtype, device=x.device)

        compute_dtype = torch.promote_types(x.dtype, self.reference_bank.dtype)
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        reference = self.reference_bank.to(dtype=compute_dtype)
        query = x.detach().to(device=self.device, dtype=compute_dtype)
        score_blocks: list[torch.Tensor] = []
        distance_fn = self.distance.distance_function
        for q_start in range(0, query.shape[0], self.distance_batch_size):
            q = query[q_start : q_start + self.distance_batch_size]
            nearest = torch.full(
                (q.shape[0],), torch.inf, dtype=compute_dtype, device=self.device
            )
            for r_start in range(0, reference.shape[0], self.distance_batch_size):
                r = reference[r_start : r_start + self.distance_batch_size]
                distances = distance_fn(q, r)
                if not torch.isfinite(distances).all():
                    raise ValueError("Computed novelty distances are not finite.")
                nearest = torch.minimum(nearest, distances.min(dim=1).values)
            score_blocks.append(nearest)
        scores = torch.cat(score_blocks)
        result = scores.to(device=x.device, dtype=x.dtype)
        if not torch.isfinite(result).all():
            raise ValueError(
                "Computed novelty scores are not finite in the input dtype."
            )
        return result


class GoldNoveltyScorer(GoldDoerWithTable):
    """Fit a fixed reference and persist nearest-reference novelty scores.

    Dataset and table rows must already contain one vector per row. Version one
    materializes the complete query source and reference in memory; `chunk` bounds
    score computation blocks, not source staging. The reference bank and fitted
    reducer are memory-only. Refitting the same fitted scorer object is rejected
    after completed scores exist, while a fresh scorer may initially fit and resume.
    The caller must provide the same reference and settings after process restart;
    Goldener cannot verify reference identity across processes. Every query chunk is
    compared against the same complete reference.

    Attributes:
        table_path: Destination table path.
        scorer: Tensor scoring tool; defaults to a fresh nearest-neighbor tool.
        reducer: Optional fitted reducer applied consistently to reference and queries.
        vectorized_key: Input vector column/key.
        score_key: Persisted novelty score column.
        chunk: Maximum query rows scored together.
        include_vectorized_in_table: Whether original query vectors are retained;
            Dataset inputs always retain vectors. Retained bfloat16 vectors are stored
            as float32 because PixelTable does not support bfloat16 arrays.
        distribute: Distributed processing capability, currently unsupported.
        collate_fn: Batch collation function, defaulting to sequence-preserving collation.
        to_keep_schema: Optional schema of extra source columns to retain.
        min_pxt_insert_size: Buffered table write size.
        batch_size: Dataset loading batch size.
        num_workers: Dataset loader worker count.
        allow_existing: Whether an existing result table may be resumed.
        drop_table: Whether the table is dropped after snapshot conversion.
        max_batches: Maximum number of pending score batches computed per call.
    """

    _MINIMAL_SCHEMA = {
        "idx": pxt.Required[pxt.Int],
        "idx_vector": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        table_path: str,
        scorer: GoldNoveltyScoringTool | None = None,
        reducer: GoldReductionToolWithFit | None = None,
        vectorized_key: str = "vectorized",
        score_key: str = "novelty",
        include_vectorized_in_table: bool = False,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        to_keep_schema: dict[str, type] | None = None,
        min_pxt_insert_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 0,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        """Configure persisted novelty scoring.

        Args:
            table_path: Local PixelTable destination.
            scorer: Tensor tool, or a fresh nearest-neighbor scorer by default.
            reducer: Optional reducer fitted once on the full reference.
            vectorized_key: Input vector column/key. Table input also requires `idx`
                and `idx_vector`; Dataset input preserves them when present, otherwise
                row positions supply `idx_vector` and then `idx`.
            score_key: Output score column.
            include_vectorized_in_table: Retain original vectors for table input.
                Dataset-backed score tables always retain vectors for persisted scoring.
                Retained bfloat16 vectors are stored as float32 because PixelTable does
                not support bfloat16 arrays; this conversion preserves represented values.
            chunk: Maximum query rows scored together, or None to use `batch_size`.
            collate_fn: Optional dataset collator.
            to_keep_schema: Optional source metadata schema.
            min_pxt_insert_size: Number of staged rows buffered before insertion.
            batch_size: Dataset loader batch size.
            num_workers: Dataset loader worker count.
            allow_existing: Whether to resume an existing result table.
            distribute: Distributed processing, which is unsupported.
            drop_table: Drop the table after `score_in_dataset` snapshots the result.
            max_batches: Maximum pending scoring batches processed per call.

        Raises:
            ValueError: If sizes are invalid, names collide, or a score column is reserved.
            NotImplementedError: If `distribute` is True.
        """
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
        if chunk is not None and (
            not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0
        ):
            raise ValueError("chunk must be a positive integer or None.")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
            or not isinstance(min_pxt_insert_size, int)
            or isinstance(min_pxt_insert_size, bool)
            or min_pxt_insert_size <= 0
        ):
            raise ValueError(
                "batch_size and min_pxt_insert_size must be positive integers."
            )
        if score_key in {"idx", "idx_vector"}:
            raise ValueError("score_key cannot be a reserved index column.")
        if score_key == vectorized_key:
            raise ValueError("score_key and vectorized_key must be different.")
        if vectorized_key in {"idx", "idx_vector"}:
            raise ValueError("vectorized_key cannot be a reserved index column.")
        if {"idx", "idx_vector"}.intersection(to_keep_schema or {}):
            raise ValueError("to_keep_schema cannot redefine reserved index columns.")
        if {score_key, vectorized_key}.intersection(to_keep_schema or {}):
            raise ValueError("to_keep_schema cannot redefine score or vector columns.")
        if (
            not isinstance(num_workers, int)
            or isinstance(num_workers, bool)
            or num_workers < 0
        ):
            raise ValueError("num_workers must be non-negative.")
        if max_batches is not None and (
            not isinstance(max_batches, int)
            or isinstance(max_batches, bool)
            or max_batches < 0
        ):
            raise ValueError("max_batches must be a non-negative integer or None.")
        self.table_path = table_path
        self.scorer = (
            scorer if scorer is not None else GoldNearestNeighborNoveltyScoringTool()
        )
        self.reducer = reducer
        self.vectorized_key = vectorized_key
        self.score_key = score_key
        self.include_vectorized_in_table = include_vectorized_in_table
        self.chunk = chunk
        if distribute:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldNoveltyScorer."
            )
        self._distribute = False

    @property
    def distribute(self) -> bool:
        """Whether distributed processing is enabled."""
        return self._distribute

    @distribute.setter
    def distribute(self, value: bool) -> None:
        """Set the unsupported distributed option.

        Raises:
            NotImplementedError: If `value` is True.
        """
        if value:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldNoveltyScorer."
            )
        self._distribute = value

    def _vectors(
        self,
        source: Dataset | Table,
        stack_vectors: bool = True,
        validate_metadata: bool = True,
    ) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        if isinstance(source, Table):
            if self.vectorized_key not in source.columns():
                raise ValueError(
                    f"Source table does not contain required column {self.vectorized_key}."
                )
            if not {"idx", "idx_vector"}.issubset(source.columns()):
                raise ValueError(
                    "Source table must contain idx and idx_vector columns."
                )
            cols = ["idx", "idx_vector", self.vectorized_key]
            cols.extend(k for k in (self.to_keep_schema or {}) if k in source.columns())
            data = GoldPxtTorchDataset(
                source.select(*[get_expr_from_column_name(source, c) for c in cols]),
                keep_cache=False,
            )
            for pos, item in enumerate(data):
                idx_vector = self._normalize_index(item.get("idx_vector", pos))
                idx = self._normalize_index(item.get("idx", idx_vector))
                item["idx"] = idx
                item["idx_vector"] = idx_vector
                missing_metadata = set(self.to_keep_schema or {}).difference(item)
                if validate_metadata and missing_metadata:
                    raise ValueError(
                        f"Source table is missing metadata columns: {missing_metadata}."
                    )
                rows.append(item)
        else:
            loader = DataLoader(
                source,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
            )
            for batch_idx, batch in enumerate(loader):
                if self.vectorized_key not in batch:
                    raise ValueError(
                        f"Dataset does not contain required key {self.vectorized_key}."
                    )
                vectors = batch[self.vectorized_key]
                if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
                    raise ValueError(
                        "Batched vectorized values must have shape "
                        "(batch_size, feature_dim)."
                    )
                for i, vector in enumerate(vectors):
                    fallback_idx = batch_idx * self.batch_size + i
                    idx_vector = (
                        batch["idx_vector"][i]
                        if "idx_vector" in batch
                        else fallback_idx
                    )
                    idx = batch["idx"][i] if "idx" in batch else idx_vector
                    one = {
                        k: (
                            v[i]
                            if isinstance(v, torch.Tensor)
                            and v.shape[:1] == vectors.shape[:1]
                            else v[i]
                            if isinstance(v, (list, tuple))
                            else v
                        )
                        for k, v in batch.items()
                    }
                    one["idx"] = self._normalize_index(idx)
                    one["idx_vector"] = self._normalize_index(idx_vector)
                    one[self.vectorized_key] = vector
                    missing_metadata = set(self.to_keep_schema or {}).difference(one)
                    if validate_metadata and missing_metadata:
                        raise ValueError(
                            f"Dataset is missing metadata keys: {missing_metadata}."
                        )
                    rows.append(one)
        if not rows:
            raise ValueError("Vector source must not be empty.")
        vectors = [torch.as_tensor(row[self.vectorized_key]) for row in rows]
        if any(v.ndim != 1 for v in vectors):
            raise ValueError("Each source row must contain one 1D vector.")
        if any(v.shape[0] == 0 for v in vectors):
            raise ValueError("Source vectors must have positive feature width.")
        if len({v.shape for v in vectors}) != 1:
            raise ValueError("All source vectors must have the same feature width.")
        for vector in vectors:
            if not vector.is_floating_point():
                raise ValueError("Source vectors must have a floating-point dtype.")
            if not torch.isfinite(vector).all():
                raise ValueError("Source vectors must contain only finite values.")
        idx_vectors = [row["idx_vector"] for row in rows]
        if len(set(idx_vectors)) != len(idx_vectors):
            raise ValueError("Source idx_vector values must be unique.")
        return (torch.stack(vectors) if stack_vectors else None), rows

    @staticmethod
    def _normalize_index(value: Any) -> int:
        """Validate and normalize a scalar integer sample or vector ID."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Sample and vector IDs must be scalar integers.")
            value = value.item()
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError("Sample and vector IDs must be integer scalars.")
        return int(value)

    @staticmethod
    def _vector_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert retained vectors to supported PixelTable arrays.

        PixelTable does not accept bfloat16 arrays, so those values are widened to
        float32 without losing their represented values. Other dtypes are preserved.
        """
        if isinstance(value, torch.Tensor):
            if value.dtype is torch.bfloat16:
                value = value.to(torch.float32)
            return value.detach().cpu().numpy()
        return value

    def fit_scorer(self, reference: Dataset | Table) -> None:
        """Fit reducer and scoring tool on the complete reference vectors.

        Args:
            reference: Dataset or table of already-vectorized reference rows.

        Raises:
            ValueError: If the reference is invalid, the reducer cannot transform,
                or completed destination scores prevent refitting.
        """
        existing = None
        try:
            existing = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            pass
        if (
            self.scorer.fitted
            and existing is not None
            and self.score_key in existing.columns()
        ):
            if (
                existing.where(
                    get_expr_from_column_name(existing, self.score_key) != None  # noqa: E711
                ).count()
                > 0
            ):  # noqa: E711
                raise ValueError(
                    "Cannot refit a scorer after this destination contains completed scores."
                )
        if self.reducer is not None and not self.reducer.supports_transform:
            raise ValueError(
                "Novelty scoring requires a reducer that supports transform."
            )
        vectors, _ = self._vectors(reference, validate_metadata=False)
        assert vectors is not None
        if self.reducer is not None:
            expected_rows = vectors.shape[0]
            self.reducer.fit(vectors)
            vectors = self.reducer.transform(vectors)
            self._validate_reduced(vectors, expected_rows)
        self.scorer.fit(vectors)

    @staticmethod
    def _validate_reduced(vectors: torch.Tensor, expected_rows: int) -> None:
        if (
            not isinstance(vectors, torch.Tensor)
            or vectors.ndim != 2
            or vectors.shape[0] != expected_rows
            or not vectors.is_floating_point()
            or not torch.isfinite(vectors).all()
        ):
            raise ValueError(
                "Reducer output must be 2D and preserve the number of rows."
            )

    def score_in_dataset(
        self, to_score: Dataset | Table, reference: Dataset | Table | None = None
    ) -> GoldPxtTorchDataset:
        """Score query vectors and return a snapshot-backed dataset.

        Args:
            to_score: Dataset or table of already-vectorized query rows.
            reference: If provided, fit the reducer and scorer before scoring. An already
                fitted scorer is refit, subject to the scored-destination guard.

        Returns:
            Snapshot-backed dataset with IDs and novelty scores.

        Raises:
            ValueError: If no fitted scorer/reference is available or input/table
                schema validation fails.
        """
        table = self.score_in_table(to_score, reference)
        dataset = GoldPxtTorchDataset(table, keep_cache=True)
        if self.drop_table:
            pxt.drop_table(table)
        return dataset

    def score_in_table(
        self, to_score: Dataset | Table, reference: Dataset | Table | None = None
    ) -> Table:
        """Persist query novelty scores, resuming rows whose score is null.

        Args:
            to_score: Dataset or table of already-vectorized query rows.
            reference: Optional full reference used to fit the scorer.

        Returns:
            PixelTable containing IDs, retained metadata, and scores.

        Raises:
            ValueError: If a scorer/reference is missing, a destination is incompatible,
                or a refit is attempted after completed scores exist.
        """
        existing = None
        try:
            existing = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            pass
        if existing is not None and not self.allow_existing:
            raise ValueError(
                f"Table at path {self.table_path} already exists and allow_existing is set to False."
            )
        _, source_rows = self._vectors(to_score, stack_vectors=False)
        keep_vectors = self.include_vectorized_in_table or isinstance(to_score, Dataset)
        if existing is not None:
            check_pxt_table_has_primary_key(existing, {"idx_vector"})
            for key in ("idx", "idx_vector"):
                expected = ColumnType.normalize_type(pxt.Int)
                actual = get_expr_from_column_name(existing, key).col_type
                if not expected.matches(actual):
                    raise ValueError("Existing ID columns must have integer types.")
            if self.score_key in existing.columns():
                expected = ColumnType.normalize_type(pxt.Float)
                actual = get_expr_from_column_name(existing, self.score_key).col_type
                if not expected.matches(actual):
                    raise ValueError(
                        "Existing score column must have a floating-point type."
                    )
            if keep_vectors and self.vectorized_key in existing.columns():
                shape = tuple(source_rows[0][self.vectorized_key].shape)
                dtype = self._vector_numpy(
                    source_rows[0][self.vectorized_key]
                ).dtype.type
                expected = ColumnType.normalize_type(pxt.Array[shape, dtype])  # type: ignore[misc]
                actual = get_expr_from_column_name(
                    existing, self.vectorized_key
                ).col_type
                if not expected.matches(actual):
                    raise ValueError(
                        "Existing vector column has an incompatible shape."
                    )
            for key, schema_type in (self.to_keep_schema or {}).items():
                if key not in existing.columns():
                    raise ValueError(
                        f"Existing result table is missing metadata column {key}."
                    )
                expected_type = ColumnType.normalize_type(schema_type)
                actual_type = get_expr_from_column_name(existing, key).col_type
                if not expected_type.matches(actual_type):
                    raise ValueError(
                        f"Existing metadata column {key} has an incompatible type."
                    )
        if reference is not None:
            self.fit_scorer(reference)
        elif not self.scorer.fitted:
            raise ValueError(
                "A fitted scorer or a reference dataset/table is required."
            )
        if self.reducer is not None and not self.reducer.supports_transform:
            raise ValueError(
                "Novelty scoring requires a reducer that supports transform."
            )

        schema = self._MINIMAL_SCHEMA | (self.to_keep_schema or {})
        table = get_valid_table(
            existing if existing is not None else self.table_path,
            schema,
            primary_key="idx_vector",
        )
        cols = table.columns()
        if self.score_key not in cols:
            table.add_column(**{self.score_key: pxt.Float})  # type: ignore[arg-type]
        if keep_vectors and self.vectorized_key not in table.columns():
            shape = tuple(source_rows[0][self.vectorized_key].shape)
            dtype = self._vector_numpy(source_rows[0][self.vectorized_key]).dtype.type
            table.add_column(**{self.vectorized_key: pxt.Array[shape, dtype]})  # type: ignore[misc]
        present = {r["idx_vector"] for r in table.select(table.idx_vector).collect()}
        inserts: list[dict[str, Any]] = []
        for row in source_rows:
            if row["idx_vector"] in present:
                continue
            data = {
                "idx": row["idx"],
                "idx_vector": row["idx_vector"],
                self.score_key: None,
            }
            for key in self.to_keep_schema or {}:
                if key in row:
                    value = row[key]
                    data[key] = (
                        value.detach().cpu().numpy()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
            if keep_vectors:
                value = row[self.vectorized_key]
                data[self.vectorized_key] = self._vector_numpy(value)
            inserts.append(data)
            present.add(row["idx_vector"])
            if len(inserts) >= self.min_pxt_insert_size:
                table.batch_update(inserts, if_not_exists="insert")
                inserts = []
        if inserts:
            table.batch_update(inserts, if_not_exists="insert")

        pending = [
            r["idx_vector"]
            for r in table.where(
                get_expr_from_column_name(table, self.score_key) == None  # noqa: E711
            )
            .select(table.idx_vector)
            .collect()
        ]  # noqa: E711
        limit = self.max_batches
        batch_size = self.chunk or self.batch_size
        processed = 0
        updates: list[dict[str, Any]] = []
        vectors_by_id = {
            r["idx_vector"]: torch.as_tensor(r[self.vectorized_key])
            for r in source_rows
        }
        for start in range(0, len(pending), batch_size):
            if limit is not None and processed >= limit:
                break
            ids = pending[start : start + batch_size]
            if any(i not in vectors_by_id for i in ids):
                raise ValueError(
                    "A pending destination vector is missing from the supplied source."
                )
            query = torch.stack([vectors_by_id[i] for i in ids])
            if self.reducer is not None:
                query = self.reducer.transform(query)
                self._validate_reduced(query, len(ids))
            scores = self.scorer.score(query)
            if (
                not isinstance(scores, torch.Tensor)
                or scores.ndim != 1
                or len(scores) != len(ids)
                or not torch.isfinite(scores).all()
            ):
                raise ValueError(
                    "Scorer output must contain one finite score per query vector."
                )
            updates.extend(
                {"idx_vector": idx, self.score_key: float(score.item())}
                for idx, score in zip(ids, scores)
            )
            if len(updates) >= self.min_pxt_insert_size:
                table.batch_update(updates)
                updates = []
            processed += 1
        if updates:
            table.batch_update(updates)
        return table

    def estimate_computation_time(self, dataset: Dataset) -> float | None:
        """Return None because generic scorer/reducer estimation is not state-safe.

        Args:
            dataset: Dataset whose scoring time would be estimated.

        Returns:
            None; arbitrary tool runtimes cannot be estimated safely.
        """
        return None

    @staticmethod
    def get_novelty_scores(
        table: Table, score_key: str = "novelty"
    ) -> dict[int, float]:
        """Get computed scores keyed by `idx_vector`, omitting null rows.

        Args:
            table: Scored novelty table.
            score_key: Score column name.

        Returns:
            Mapping from vector IDs to computed scores.
        """
        return {
            int(r["idx_vector"]): float(r[score_key])
            for r in table.select(
                table.idx_vector, get_expr_from_column_name(table, score_key)
            ).collect()
            if r[score_key] is not None
        }

    @staticmethod
    def get_novel_indices(
        table: Table, threshold: float, idx_key: str = "idx", score_key: str = "novelty"
    ) -> set[int]:
        """Return distinct sample/vector IDs with scores strictly above threshold.

        Args:
            table: Scored novelty table.
            threshold: Finite distance threshold using strict `score > threshold`.
            idx_key: Either sample `idx` or individual `idx_vector`.
            score_key: Score column name.

        Returns:
            Set of matching IDs. With `idx_key="idx"`, any novel vector marks its sample.

        Raises:
            ValueError: If the index key is unsupported or the threshold is not finite.
        """
        if idx_key not in ("idx", "idx_vector"):
            raise ValueError("idx_key must be either 'idx' or 'idx_vector'.")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        idx = get_expr_from_column_name(table, idx_key)
        score = get_expr_from_column_name(table, score_key)
        return {
            int(row[idx_key])
            for row in table.where(score > threshold).select(idx).distinct().collect()
        }
