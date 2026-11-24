"""Score vectorized samples for novelty relative to a reference population.

Use this stage to score query vectors against a fixed reference population.
Tensor tools compute scores and dataset and table orchestration persist them.

- `GoldNoveltyScoringTool` defines the tensor interface for fitting on reference
  vectors and scoring queries.
- `GoldNearestNeighborNoveltyScoringTool` scores each query by its minimum
  distance to the fitted reference vectors.
- `GoldNoveltyScorer` fits the reference and persists raw novelty scores per
  vector. It retrieves scores by vector or by sample, using the highest vector
  score for each sample.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable, Literal

import numpy as np
import pixeltable as pxt
import torch
from pixeltable import Error
from pixeltable.catalog import Table
from pixeltable.type_system import ColumnType
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from goldener.distance import DistanceType
from goldener.do import GoldDoerWithTable
from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    check_pxt_table_has_primary_key,
    get_expr_from_column_name,
    get_valid_table,
    make_batch_ready_for_table,
)
from goldener.reduce import GoldReductionToolWithFit
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import (
    filter_batch_from_indices,
    validate_positive_integer,
    validate_vectorized_input,
)

logger = getLogger(__name__)


class GoldNoveltyScoringTool(ABC):
    """Score vectors for novelty using a tool fitted on reference data.

    Inputs have shape `(num_vectors, feature_dim)` and scores have shape
    `(num_vectors,)`. Higher scores indicate greater novelty. I

    Implementations define the meaning and scale of their scores, such as nearest-reference
    distance or predicted probability of novelty.

    Fitted state is kept in memory.

    Attributes:
        device: Device used for computation. Defaults to CPU.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the GoldNoveltyScoringTool.

        Args:
            device: Device used for computation. Defaults to CPU.
        """
        self.device = torch.device(device)

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Whether a reference has been fitted.

        Returns:
            True when this tool is ready to score query vectors.
        """

    @abstractmethod
    def fit(self, x: torch.Tensor) -> None:
        """Fit the scoring tool on 2D reference vectors.

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
            A 1D novelty scoring tensor on the input device.

        Raises:
            ValueError: If the tool is unfitted or inputs do not satisfy the tensor contract.
        """


class GoldNearestNeighborNoveltyScoringTool(GoldNoveltyScoringTool):
    """Novelty scoring from the minimum distance to any fitted reference vector.

    Exact computation stores the complete reference bank in memory.

    Pairwise temporary allocations are bounded by `distance_batch_size` query and reference
    rows.

    Attributes:
        distance: Pairwise distance metric. Defaults to cosine.
        distance_batch_size: Maximum rows in each query or reference computation
            block. Defaults to 1024.
        device: Device used for tensor computation. Defaults to CPU.
    """

    def __init__(
        self,
        distance: DistanceType = DistanceType.COSINE,
        distance_batch_size: int = 1024,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the GoldNearestNeighborNoveltyScoringTool.

        Args:
            distance: Pairwise distance selector. Defaults to cosine.
            device: Computation device. Defaults to CPU.
            distance_batch_size: Positive query and reference block size. Defaults to 1024.

        Raises:
            ValueError: If `distance_batch_size` is not a positive integer.
        """
        super().__init__(device=device)
        validate_positive_integer(distance_batch_size, "distance_batch_size")
        self.distance = distance
        self.distance_batch_size = distance_batch_size
        self._reference_bank: torch.Tensor | None = None

    @property
    def fitted(self) -> bool:
        """Whether a reference bank is available."""
        return self._reference_bank is not None

    def fit(self, x: torch.Tensor) -> None:
        """Replace the fitted reference bank with an independent clone of `x`.

        Args:
            x: Nonempty finite floating reference matrix `(num_vectors, feature_dim)`.

        Raises:
            ValueError: If `x` is empty or violates the 2D floating tensor contract.
        """
        validate_vectorized_input(x, "GoldNoveltyScoringTool")
        if x.shape[0] == 0:
            raise ValueError("The novelty reference must not be empty.")
        self._reference_bank = x.detach().to(self.device).clone()

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
        if not self.fitted:
            raise ValueError("Fit the novelty scorer before calling score.")
        assert self._reference_bank is not None
        validate_vectorized_input(x, "GoldNoveltyScoringTool")

        if x.shape[1] != self._reference_bank.shape[1]:
            raise ValueError(
                "Query feature_dim must match the fitted reference feature_dim."
            )
        if x.shape[0] == 0:
            return torch.empty((0,), dtype=x.dtype, device=x.device)

        compute_dtype = torch.promote_types(x.dtype, self._reference_bank.dtype)
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        reference = self._reference_bank.to(dtype=compute_dtype)
        queries = x.detach().to(device=self.device, dtype=compute_dtype)
        reference_batches = reference.split(self.distance_batch_size)
        score_batches = []
        distance_func = self.distance.distance_function
        for query_batch in queries.split(self.distance_batch_size):
            nearest = query_batch.new_full((len(query_batch),), torch.inf)
            for reference_batch in reference_batches:
                distances = distance_func(query_batch, reference_batch)
                if not torch.isfinite(distances).all():
                    raise ValueError("Computed novelty distances are not finite.")
                nearest = torch.minimum(nearest, distances.amin(dim=1))
            score_batches.append(nearest)

        result = torch.cat(score_batches).to(x)
        if not torch.isfinite(result).all():
            raise ValueError(
                "Computed novelty scores are not finite in the input dtype."
            )
        return result


class GoldNoveltyScorer(GoldDoerWithTable):
    """Fit a fixed reference and persist nearest-reference novelty scores.

    The GoldNoveltyScorer processes a dataset or PixelTable table to score already
    vectorized representations using a novelty scoring tool. Results are stored in
    a local PixelTable table (specified by `table_path`) so that repeated calls reuse
    completed scores without duplicating or recomputing them.

    The scorer stages query vectors in the result table, fits the reference if supplied,
    and scores pending vectors in chunks against the complete reference. The reference
    bank and fitted reducer are kept in memory. Refitting the scorer is rejected
    once the table has scores. On restart, fit a fresh scorer with the same
    reference and settings before resuming.

    Scoring operates in sequential (single-process) mode; distributed processing is
    not implemented. The table schema is created or validated automatically and
    includes `idx` and `idx_vector` to link scores to source samples and vectors.

    Attributes:
        table_path: Path to the PixelTable table where scoring results will be stored locally.
        scorer: The GoldNoveltyScoringTool implementing the scoring algorithm.
            Defaults to a new GoldNearestNeighborNoveltyScoringTool.
        reducer: Optional GoldReductionToolWithFit for dimensionality reduction.
            Fitted once on the reference and applied to query vectors without refitting.
        vectorized_key: Key containing vectorized data for scoring. Defaults to "vectorized".
        score_key: Column name for novelty scores. Defaults to "novelty".
        chunk: Positive integer maximum number of query vectors scored together.
            Defaults to None, which uses `batch_size`.
        include_vectorized_in_table: Whether original query vectors are retained;
            Dataset inputs always retain vectors. Retained bfloat16 vectors are stored
            and scored as float32 because PixelTable does not support bfloat16 arrays.
        distribute: Whether to use distributed processing. Not implemented yet. Defaults to False.
        collate_fn: Optional collate function for the DataLoader. If None,
            `collate_keeping_sequences_as_sequences` is used.
        to_keep_schema: Optional dictionary defining additional columns to keep from
            the source dataset/table. Values are PixelTable types. Identity,
            vector, and score columns are managed by the scorer.
        min_pxt_insert_size: Minimum number of source rows to accumulate before inserting into PixelTable.
            Defaults to 100.
        batch_size: Batch size used for dataset ingestion. Defaults to 1.
        num_workers: Number of workers for the PyTorch DataLoader. Defaults to 0.
        allow_existing: Whether to allow using an existing table. Defaults to True.
        drop_table: Whether to drop the scoring table after creating the dataset when
            using `score_in_dataset`. Defaults to False.
        max_batches: Nonnegative integer limit on pending score batches per call.
            Defaults to None, which scores all pending batches.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
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
        """Initialize the GoldNoveltyScorer.

        Args:
            table_path: Path to store the PixelTable table.
            scorer: The GoldNoveltyScoringTool implementing the scoring algorithm.
                Defaults to a new GoldNearestNeighborNoveltyScoringTool.
            reducer: Optional GoldReductionToolWithFit for dimensionality reduction
                before scoring. Fitted once on the full reference.
            vectorized_key: Input vector column/key. Table input also requires `idx`
                and `idx_vector`; Dataset input preserves them when present, otherwise
                row positions supply `idx_vector` and then `idx`.
            score_key: Output score column.
            include_vectorized_in_table: Retain original vectors for table input.
                Dataset-backed score tables always retain vectors for persisted scoring.
                Retained bfloat16 vectors are stored as float32 because PixelTable does
                not support bfloat16 arrays. Scoring uses the stored vector dtype.
            chunk: Positive integer maximum number of query rows scored together.
                Defaults to None, which uses `batch_size`.
            collate_fn: Optional collate function for the DataLoader.
                If None, `collate_keeping_sequences_as_sequences` is used.
            to_keep_schema: Optional schema for additional columns to preserve.
            min_pxt_insert_size: Positive integer minimum number of source rows to
                accumulate before inserting into PixelTable. Defaults to 100.
            batch_size: Positive integer batch size used for dataset ingestion. Defaults to 1.
            num_workers: Nonnegative integer number of workers for the PyTorch DataLoader.
                Defaults to 0.
            allow_existing: Whether to allow using an existing table. Defaults to True.
            distribute: Whether to use distributed scoring. Not implemented yet. Defaults to False.
            drop_table: Whether to drop the table after dataset creation when using
                `score_in_dataset`. Defaults to False.
            max_batches: Nonnegative integer limit on pending score batches per call.
                Defaults to None, which scores all pending batches.

        Raises:
            ValueError: If a numeric option is outside its allowed range or is not an
                integer, two column keys share a name, a column key uses an identity
                name, or `to_keep_schema` redefines an identity, vector, or score column.
            NotImplementedError: If `distribute` is True.
        """
        validate_positive_integer(chunk, "chunk", allow_none=True)
        validate_positive_integer(min_pxt_insert_size, "min_pxt_insert_size")
        validate_positive_integer(batch_size, "batch_size")
        validate_positive_integer(num_workers, "num_workers", allow_zero=True)
        validate_positive_integer(
            max_batches, "max_batches", allow_zero=True, allow_none=True
        )

        self.scorer = (
            scorer if scorer is not None else GoldNearestNeighborNoveltyScoringTool()
        )
        self.reducer = reducer
        self.chunk = chunk
        self.vectorized_key = vectorized_key
        self.include_vectorized_in_table = include_vectorized_in_table
        self.score_key = score_key
        self.distribute = distribute

        super().__init__(
            table_path=table_path,
            allow_existing=allow_existing,
            drop_table=drop_table,
            max_batches=max_batches,
            collate_fn=collate_fn,
            to_keep_schema=to_keep_schema,
            min_pxt_insert_size=min_pxt_insert_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if {self.score_key, self.vectorized_key}.intersection(
            self.to_keep_schema or {}
        ):
            raise ValueError("to_keep_schema cannot redefine score or vector columns.")

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
                "Distributed processing is not implemented for GoldNoveltyScorer."
            )
        self._distribute = False

    def score_in_table(
        self, to_score: Dataset | Table, reference: Dataset | Table | None = None
    ) -> Table:
        """Score vectorized samples and store results in a PixelTable table.

        The scoring process compares already vectorized representations with a fitted
        reference and stores results in the table specified by `table_path`. If a
        reducer is provided, queries use the same transformation as the reference.

        This method is idempotent: repeated calls reuse completed scores. Inputs,
        reference vectors, and scoring settings must remain unchanged when resuming.
        `max_batches` limits pending score batches per call; all source rows are staged.

        Args:
            to_score: Dataset or Table to score. Dataset items must contain a 1D
                floating vector under `vectorized_key`; collation must produce a
                2D tensor shaped `(batch_size, feature_dim)`. Missing
                `idx_vector` values default to row positions, and missing `idx`
                values default to vector IDs. Table inputs require `vectorized_key`,
                `idx`, and `idx_vector` columns. Vector IDs must be unique.
            reference: Optional dataset or table used to fit the scorer and reducer.

        Returns:
            A PixelTable Table containing scores in `score_key`, sample indices in
            `idx`, vector indices in `idx_vector`, and configured retained columns.
            Unprocessed vectors have null scores.

        Raises:
            ValueError: If a scorer/reference is missing, inputs or the destination
                schema are invalid, or refitting would change completed scores.
        """
        if reference is None and not self.scorer.fitted:
            raise ValueError(
                "A fitted scorer or a reference dataset/table is required."
            )
        if self.reducer is not None and not self.reducer.supports_transform:
            raise ValueError(
                "Novelty scoring requires a reducer that supports transform."
            )

        scoring_table = self.setup_scoring_table(to_score)
        if reference is not None:
            self.fit_scorer(reference)
        source_table = to_score if isinstance(to_score, Table) else scoring_table
        if self.distribute:
            self._distributed_score(scoring_table, source_table)
        else:
            self._sequential_score(scoring_table, source_table)
        return scoring_table

    def score_in_dataset(
        self, to_score: Dataset | Table, reference: Dataset | Table | None = None
    ) -> GoldPxtTorchDataset:
        """Score vectorized samples and return results as a GoldPxtTorchDataset.

        This is a convenience wrapper that runs `score_in_table` to populate or
        resume the scoring table, then wraps the results in a GoldPxtTorchDataset.
        If `drop_table` is True, the table is removed after the dataset is created.
        The cached dataset remains usable after deletion and contains raw per-vector
        scores. Use `get_novelty_scores` to retrieve scores by vector or sample ID.

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
        scored_table = self.score_in_table(to_score, reference)

        dataset = GoldPxtTorchDataset(scored_table, keep_cache=True)

        if self.drop_table:
            pxt.drop_table(self.table_path)

        return dataset

    def setup_scoring_table(self, to_score: Dataset | Table) -> Table:
        """Set up the scoring table based on the source dataset or table.

        Args:
            to_score: Dataset or Table containing already vectorized samples.
                Missing dataset indices are generated from row positions.

        Returns:
            A PixelTable Table populated with source indices and retained columns.
            Existing scores are preserved and new rows have null scores. This
            method stages rows without running the scorer.

        Raises:
            ValueError: If `allow_existing` is False and the table already exists,
                or required columns, vectors, indices, or stored schema are invalid.
        """
        logger.info(f"Loading the existing scoring table from {self.table_path}")
        try:
            old_scoring_table = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            logger.info(f"No existing scoring table from {self.table_path}")
            old_scoring_table = None

        if old_scoring_table is not None:
            if not self.allow_existing:
                raise ValueError(
                    f"Table at path {self.table_path} already exists and allow_existing is set to False."
                )
            check_pxt_table_has_primary_key(old_scoring_table, {"idx_vector"})
            scoring_table_cols = set(old_scoring_table.columns())
            expected_schema = self._MINIMAL_SCHEMA | (self.to_keep_schema or {})
            if self.score_key in scoring_table_cols:
                expected_schema[self.score_key] = pxt.Float
            for key, schema_type in expected_schema.items():
                if key not in scoring_table_cols:
                    raise ValueError(f"Existing result table is missing column {key}.")
                expected = ColumnType.normalize_type(schema_type)
                actual = get_expr_from_column_name(old_scoring_table, key).col_type
                if not expected.matches(actual):
                    raise ValueError(f"Existing column {key} has an incompatible type.")

        if isinstance(to_score, Table):
            scoring_table = self._scoring_table_from_table(
                score_from=to_score,
                old_scoring_table=old_scoring_table,
            )
        else:
            scoring_table = self._scoring_table_from_dataset(
                to_score, old_scoring_table
            )
        return scoring_table

    def _scoring_table_from_table(
        self, score_from: Table, old_scoring_table: Table | None
    ) -> Table:
        """Create or validate the scoring table schema from a PixelTable table.

        Args:
            score_from: The source PixelTable table with query vectors.
            old_scoring_table: Existing result table when resuming, or None.

        Returns:
            The scoring table populated with source rows and any existing scores.

        Raises:
            ValueError: If a required column is missing, a vector column has an
                incompatible shape, or source rows fail validation.
        """
        required_columns = {"idx", "idx_vector", self.vectorized_key}
        required_columns.update(self.to_keep_schema or {})
        if missing := required_columns.difference(score_from.columns()):
            raise ValueError(f"Source table is missing required columns: {missing}.")

        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema
        scoring_table = get_valid_table(
            table=old_scoring_table
            if old_scoring_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )
        scoring_table_cols = scoring_table.columns()
        if self.score_key not in scoring_table_cols:
            scoring_table.add_column(if_exists="error", **{self.score_key: pxt.Float})

        if self.include_vectorized_in_table:
            vector_type = get_expr_from_column_name(
                score_from, self.vectorized_key
            ).col_type
            if self.vectorized_key not in scoring_table_cols:
                scoring_table.add_column(
                    if_exists="error",
                    **{self.vectorized_key: vector_type},  # type: ignore[arg-type]
                )
            elif not vector_type.matches(
                get_expr_from_column_name(scoring_table, self.vectorized_key).col_type
            ):
                raise ValueError("Existing vector column has an incompatible shape.")

        col_list = ["idx", "idx_vector", self.vectorized_key]
        if self.to_keep_schema is not None:
            col_list.extend(self.to_keep_schema)
        self._add_rows_to_scoring_table_from_dataset(
            score_from=GoldPxtTorchDataset(
                score_from.select(
                    *[get_expr_from_column_name(score_from, col) for col in col_list]
                ),
                keep_cache=False,
            ),
            scoring_table=scoring_table,
            include_vectorized=self.include_vectorized_in_table,
        )
        return scoring_table

    def _scoring_table_from_dataset(
        self, score_from: Dataset, old_scoring_table: Table | None
    ) -> Table:
        """Create or validate the scoring table schema from a PyTorch Dataset.

        Args:
            score_from: The source dataset with query vectors.
            old_scoring_table: Existing result table when resuming, or None.

        Returns:
            The scoring table populated with source rows and any existing scores.
            Dataset inputs always retain their vectors in the result table.

        Raises:
            ValueError: If the dataset is empty, its vectors or identities are
                invalid, or the result table has an incompatible schema.
        """
        if hasattr(score_from, "__len__") and len(score_from) == 0:
            raise ValueError("Vector source must not be empty.")
        sample = get_dataset_sample_dict(
            score_from, collate_fn=self.collate_fn, expected=[self.vectorized_key]
        )
        sample_vectors = sample[self.vectorized_key]
        validate_vectorized_input(sample_vectors, "GoldNoveltyScoringTool")
        sample_vector = sample_vectors[0]
        vector_dtype = (
            np.float32
            if sample_vector.dtype == torch.bfloat16
            else sample_vector.detach().cpu().numpy().dtype.type
        )
        vector_type: type = pxt.Array[tuple(sample_vector.shape), vector_dtype]  # type: ignore[misc]

        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema
        scoring_table = get_valid_table(
            table=old_scoring_table
            if old_scoring_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )
        scoring_table_cols = scoring_table.columns()
        if self.vectorized_key not in scoring_table_cols:
            scoring_table.add_column(
                if_exists="error", **{self.vectorized_key: vector_type}
            )
        elif not ColumnType.normalize_type(vector_type).matches(
            get_expr_from_column_name(scoring_table, self.vectorized_key).col_type
        ):
            raise ValueError("Existing vector column has an incompatible shape.")
        if self.score_key not in scoring_table_cols:
            scoring_table.add_column(if_exists="error", **{self.score_key: pxt.Float})

        self._add_rows_to_scoring_table_from_dataset(
            score_from=score_from,
            scoring_table=scoring_table,
            include_vectorized=True,
        )
        return scoring_table

    def _add_rows_to_scoring_table_from_dataset(
        self,
        score_from: Dataset,
        scoring_table: Table,
        include_vectorized: bool = False,
    ) -> None:
        """Populate the scoring table in batches, preserving completed rows.

        Missing vector IDs default to row positions; missing sample IDs default to
        vector IDs. All rows are ingested before scoring, independently of `max_batches`.

        Args:
            score_from: The source dataset with query vectors.
            scoring_table: The result table to populate.
            include_vectorized: Whether to copy the vectors into the result table.

        Raises:
            ValueError: If vectors or required metadata are missing or invalid,
                or a source vector ID occurs more than once.
        """
        dataloader = DataLoader(
            score_from,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        to_exclude_from_batch = {
            int(row["idx_vector"])
            for row in scoring_table.select(scoring_table.idx_vector).collect()
        }
        seen_indices: set[int] = set()
        feature_dim = None
        ready_to_insert: list[dict[str, Any]] = []
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Initializing rows for the scoring table",
            total=len(dataloader) if hasattr(score_from, "__len__") else None,
        ):
            if self.vectorized_key not in batch:
                raise ValueError(
                    f"Dataset does not contain required key {self.vectorized_key}."
                )
            vectors = batch[self.vectorized_key]
            validate_vectorized_input(vectors, "GoldNoveltyScoringTool")
            if feature_dim is not None and vectors.shape[1] != feature_dim:
                raise ValueError("All source vectors must have the same feature width.")
            feature_dim = vectors.shape[1]
            if missing := set(self.to_keep_schema or {}).difference(batch):
                raise ValueError(f"Dataset is missing metadata keys: {missing}.")
            if "idx_vector" not in batch:
                start = batch_idx * self.batch_size
                batch["idx_vector"] = list(range(start, start + len(vectors)))
            if "idx" not in batch:
                batch["idx"] = batch["idx_vector"]
            for key in ("idx", "idx_vector"):
                normalized_indices: list[int] = []
                for value in batch[key]:
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            raise ValueError(
                                "Sample and vector IDs must be scalar integers."
                            )
                        value = value.item()
                    if isinstance(value, bool) or not isinstance(
                        value, (int, np.integer)
                    ):
                        raise ValueError(
                            "Sample and vector IDs must be integer scalars."
                        )
                    normalized_indices.append(int(value))
                batch[key] = normalized_indices
            indices = batch["idx_vector"]
            if len(set(indices)) != len(indices) or seen_indices.intersection(indices):
                raise ValueError("Source idx_vector values must be unique.")
            seen_indices.update(indices)

            if to_exclude_from_batch:
                batch = filter_batch_from_indices(
                    batch, to_exclude_from_batch, index_key="idx_vector"
                )
                if not batch:
                    continue

            batch[self.score_key] = [None] * len(batch["idx_vector"])
            to_insert_keys = ["idx", self.score_key]
            if self.to_keep_schema is not None:
                to_insert_keys.extend(self.to_keep_schema)
                for key in self.to_keep_schema:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].detach().cpu().numpy()
            if include_vectorized:
                to_insert_keys.append(self.vectorized_key)
                vectors_to_store = batch[self.vectorized_key]
                if vectors_to_store.dtype == torch.bfloat16:
                    vectors_to_store = vectors_to_store.to(torch.float32)
                batch[self.vectorized_key] = vectors_to_store.detach().cpu().numpy()
            batch_as_list = make_batch_ready_for_table(
                batch, to_insert_keys, "idx_vector"
            )
            ready_to_insert.extend(batch_as_list)
            if len(ready_to_insert) >= self.min_pxt_insert_size:
                scoring_table.insert(ready_to_insert)
                ready_to_insert = []
        if ready_to_insert:
            scoring_table.insert(ready_to_insert)
        if not seen_indices:
            raise ValueError("Vector source must not be empty.")

    def _distributed_score(self, scoring_table: Table, to_score: Table) -> None:
        """Reject distributed scoring.

        Args:
            scoring_table: The table in which to store novelty scores.
            to_score: The source table containing query vectors.

        Raises:
            NotImplementedError: Distributed scoring is unavailable.
        """
        raise NotImplementedError("Distributed novelty scoring is not implemented.")

    def _sequential_score(self, scoring_table: Table, to_score: Table) -> None:
        """Score pending query chunks against the same fitted reference.

        Each completed chunk is written to PixelTable by vector ID.

        Args:
            scoring_table: The table in which to store novelty scores.
            to_score: The source table containing query vectors.

        Raises:
            ValueError: If a pending vector is missing or scoring returns invalid data.
        """
        score_col = get_expr_from_column_name(scoring_table, self.score_key)
        scoring_pool = scoring_table.where(score_col == None)  # noqa: E711
        pending_vector_indices = [
            int(row["idx_vector"])
            for row in scoring_pool.select(scoring_table.idx_vector).collect()
        ]
        if not pending_vector_indices:
            return

        vector_table = scoring_table if self.include_vectorized_in_table else to_score
        vector_col = get_expr_from_column_name(vector_table, self.vectorized_key)
        batch_size = self.chunk or self.batch_size

        for chunk_id, start in enumerate(
            range(0, len(pending_vector_indices), batch_size)
        ):
            if self.max_batches is not None and chunk_id >= self.max_batches:
                break
            chunk_vector_indices = pending_vector_indices[start : start + batch_size]
            rows = (
                vector_table.where(vector_table.idx_vector.isin(chunk_vector_indices))
                .select(vector_table.idx_vector, vector_col)
                .collect()
            )
            if len(rows) != len(chunk_vector_indices):
                raise ValueError(
                    "A pending destination vector is missing from the supplied source."
                )

            indices = [int(row["idx_vector"]) for row in rows]
            vectors = torch.stack(
                [torch.as_tensor(row[self.vectorized_key]) for row in rows]
            )
            scores = self._compute_scores(vectors)
            scoring_table.batch_update(
                [
                    {"idx_vector": idx, self.score_key: float(score.item())}
                    for idx, score in zip(indices, scores, strict=True)
                ]
            )
            logger.info(f"Scored {len(indices)} vectors from chunk {chunk_id}")

    def _compute_scores(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the fitted reducer and score query vectors in their input order.

        Args:
            vectors: Query vectors shaped `(num_vectors, feature_dim)`.

        Returns:
            A tensor of shape `(num_vectors,)` containing one score per query.

        Raises:
            ValueError: If the reducer changes the row count or returns invalid
                vectors, or the scorer fails to return one finite score per query.
        """
        if self.reducer is not None:
            expected_rows = len(vectors)
            vectors = self.reducer.transform(vectors)
            self._validate_reduced(vectors, expected_rows)

        scores = self.scorer.score(vectors)
        if (
            not isinstance(scores, torch.Tensor)
            or scores.shape != (len(vectors),)
            or not torch.isfinite(scores).all()
        ):
            raise ValueError(
                "Scorer output must contain one finite score per query vector."
            )
        return scores

    def fit_scorer(self, reference: Dataset | Table) -> None:
        """Fit reducer and scoring tool on the complete reference vectors.

        Args:
            reference: Dataset or table of already-vectorized reference rows.

        Raises:
            ValueError: If the reference is invalid, the reducer cannot transform,
                or completed destination scores prevent refitting.
        """
        try:
            existing = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            existing = None
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
            ):
                raise ValueError(
                    "Cannot refit a scorer after this destination contains completed scores."
                )
        if self.reducer is not None and not self.reducer.supports_transform:
            raise ValueError(
                "Novelty scoring requires a reducer that supports transform."
            )
        if isinstance(reference, Table):
            reference = GoldPxtTorchDataset(
                reference.select(
                    get_expr_from_column_name(reference, self.vectorized_key)
                ),
                keep_cache=False,
            )
        dataloader = DataLoader(
            reference,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        batches = []
        for batch in dataloader:
            if self.vectorized_key not in batch:
                raise ValueError(
                    f"Dataset does not contain required key {self.vectorized_key}."
                )
            validate_vectorized_input(
                batch[self.vectorized_key], "GoldNoveltyScoringTool"
            )
            batches.append(batch[self.vectorized_key])
        if not batches:
            raise ValueError("The novelty reference must not be empty.")
        vectors = torch.cat(batches)
        if self.reducer is not None:
            expected_rows = vectors.shape[0]
            self.reducer.fit(vectors)
            vectors = self.reducer.transform(vectors)
            self._validate_reduced(vectors, expected_rows)
        self.scorer.fit(vectors)

    @staticmethod
    def _validate_reduced(vectors: torch.Tensor, expected_rows: int) -> None:
        """Check that reduction returned one finite floating vector per input row."""
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

    @staticmethod
    def get_novelty_scores(
        table: Table,
        score_key: str = "novelty",
        *,
        idx_key: Literal["idx", "idx_vector"] = "idx_vector",
    ) -> dict[int, float]:
        """Retrieve computed novelty scores by vector or sample ID.

        Args:
            table: PixelTable table with raw per-vector scores. Pending rows are
                omitted.
            score_key: Score column name. Defaults to "novelty".
            idx_key: "idx_vector" returns individual vector scores (the default).
                "idx" returns the maximum available vector score for each sample.

        Returns:
            A dictionary mapping the requested IDs to their novelty scores.

        Raises:
            ValueError: If the index key is unsupported.
        """
        if idx_key not in ("idx", "idx_vector"):
            raise ValueError("idx_key must be either 'idx' or 'idx_vector'.")
        idx_col = get_expr_from_column_name(table, idx_key)
        score_col = get_expr_from_column_name(table, score_key)
        rows = table.where(score_col != None).select(idx_col, score_col).collect()  # noqa: E711

        if idx_key == "idx_vector":
            return {int(row[idx_key]): float(row[score_key]) for row in rows}

        scores: dict[int, float] = {}
        for row in rows:
            idx = int(row[idx_key])
            score = float(row[score_key])
            scores[idx] = max(scores.get(idx, score), score)
        return scores
