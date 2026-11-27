"""Score vectorized samples for novelty relative to a reference population.

Use this stage to score query vectors against a fixed reference population.
Tensor tools compute scores and dataset and table orchestration persist them.

- `GoldNoveltyScoringTool` defines the tensor interface for fitting on reference
  vectors and scoring queries.
- `GoldNearestNeighborNoveltyScoringTool` scores each query by its minimum
  distance to the fitted reference vectors.
- `GoldNoveltyScorer` persists raw novelty scores per
  vector. It retrieves scores by vector or by sample, using the highest vector
  score for each sample by default.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from logging import getLogger
from typing import Any, Callable, Literal

import pixeltable as pxt
import torch
from pixeltable import Error
from pixeltable.catalog import Table
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
    get_sample_row_from_idx,
)
from goldener.reduce import GoldReductionTool
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import (
    filter_batch_from_indices,
    validate_positive_integer,
    validate_vectorized_input,
    validate_floating_tensor,
)

logger = getLogger(__name__)


class GoldNoveltyScoringTool(ABC):
    """Score vectors for novelty using a tool fitted on reference data.

    Inputs have shape `(num_vectors, feature_dim)` and scores have shape
    `(num_vectors,)`. Higher scores indicate greater novelty.

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
        self._reference_bank = x.detach().to(self.device).clone().to(torch.float32)

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

        if x.shape[0] == 0:
            raise ValueError("The novelty query must not be empty.")
        if x.shape[1] != self._reference_bank.shape[1]:
            raise ValueError(
                "Query feature_dim must match the fitted reference feature_dim."
            )

        reference = self._reference_bank
        queries = x.detach().to(device=self.device, dtype=torch.float32)

        reference_batches = reference.split(self.distance_batch_size)
        score_batches = []
        distance_func = self.distance.distance_function
        for query_batch in queries.split(self.distance_batch_size):
            # initial to the minimum possible distance (first batch set the first real value)
            nearest = query_batch.new_full((len(query_batch),), torch.inf)
            for reference_batch in reference_batches:
                distances = distance_func(query_batch, reference_batch)
                if not torch.isfinite(distances).all():
                    raise ValueError("Computed novelty distances are not finite.")
                nearest = torch.minimum(nearest, distances.amin(dim=1))
            score_batches.append(nearest)

        result = torch.cat(score_batches).to(x.device)
        if not torch.isfinite(result).all():
            raise ValueError(
                "Computed novelty scores are not finite in the input dtype."
            )
        return result


class GoldNoveltyScorer(GoldDoerWithTable):
    """Persist novelty scores for vectorized samples.

    The GoldNoveltyScorer processes a dataset or PixelTable table to score already
    vectorized representations using a novelty scoring tool. Results are stored in
    a local PixelTable table (specified by `table_path`) so that repeated calls reuse
    completed scores without duplicating or recomputing them.

    Scoring operates in sequential (single-process) mode; distributed processing is
    not implemented. The table schema is created or validated automatically and
    includes `idx` and `idx_vector` to link scores to source samples and vectors.

    Attributes:
        table_path: Path to the PixelTable table where scoring results will be stored locally.
        scorer: Dictionary of scoring tools. A single supplied tool is stored under
            "all_labels"; a supplied dictionary maps each label to its tool.
        label_key: Optional column used to group vectors by label. Defaults to None.
        reducer: Optional GoldReductionTool for dimensionality reduction.
            Applied to query vectors before scoring.
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
        max_batches: Nonnegative integer limit on pending score batches per call
            across all labels.
            Defaults to None, which scores all pending batches.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
        "idx_vector": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        table_path: str,
        scorer: GoldNoveltyScoringTool | dict[str, GoldNoveltyScoringTool],
        reducer: GoldReductionTool | None = None,
        vectorized_key: str = "vectorized",
        score_key: str = "novelty",
        label_key: str | None = None,
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
            scorer: Scoring tool shared by all labels, or a dictionary mapping each
                label to its scoring tool. A dictionary requires `label_key`.
            reducer: Optional GoldReductionTool for dimensionality reduction
                before scoring.
            vectorized_key: Input vector column/key. Table input also requires `idx`
                and `idx_vector`; Dataset input preserves them when present, otherwise
                row positions supply `idx_vector` and then `idx`.
            score_key: Output score column.
            label_key: Optional label column for grouping query vectors. Each label
                uses its dictionary entry, or the shared tool when one is supplied.
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
            max_batches: Nonnegative integer limit on pending score batches per call
            across all labels.
                Defaults to None, which scores all pending batches.

        Raises:
            ValueError: If a numeric option is outside its allowed range or is not an
                integer, two column keys share a name, a column key uses an identity
                name, or `to_keep_schema` redefines an identity, vector, or score column.
            NotImplementedError: If `distribute` is True.
        """
        if isinstance(scorer, dict) and label_key is None:
            raise ValueError(
                "A label_key must be provided when using a scorer per label."
            )
        self.is_scorer_per_label = isinstance(scorer, dict)
        self.scorer = scorer if isinstance(scorer, dict) else {"all_labels": scorer}
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
                "Distributed processing is not implemented for GoldNoveltyScorer."
            )
        self._distribute = False

    def score_in_table(self, to_score: Dataset | Table) -> Table:
        """Score vectorized samples and store results in a PixelTable table.

        The scoring tool processes already vectorized representations and stores
        results in the table specified by `table_path`. An optional reducer
        transforms vectors before scoring.

        This method is idempotent: repeated calls reuse completed scores. Inputs,
        tool state, and scoring settings must remain unchanged when resuming.
        `max_batches` limits pending score batches per call; all source rows are staged.

        Args:
            to_score: Dataset or Table to score. Dataset items must contain a 1D
                floating vector under `vectorized_key`; collation must produce a
                2D tensor shaped `(batch_size, feature_dim)`. Missing
                `idx_vector` values default to row positions, and missing `idx`
                values default to vector IDs. Table inputs require `vectorized_key`,
                `idx`, and `idx_vector` columns. Vector IDs must be unique.

        Returns:
            A PixelTable Table containing scores in `score_key`, sample indices in
            `idx`, vector indices in `idx_vector`, and configured retained columns.
            Unprocessed vectors have null scores.

        Raises:
            ValueError: If inputs or the destination schema are invalid, or the
                reducer does not support transformation.
        """
        if self.reducer is not None and not self.reducer.supports_transform:
            raise ValueError(
                "Novelty scoring requires a reducer that supports transform."
            )

        for scorer in self.scorer.values():
            if not scorer.fitted:
                raise ValueError(
                    "All novelty scorers must be fitted before scoring query vectors."
                )

        scoring_table = self.setup_scoring_table(to_score)

        source_table = to_score if isinstance(to_score, Table) else scoring_table

        if self.distribute:
            self._distributed_score(scoring_table=scoring_table, to_score=source_table)
        else:
            self._sequential_score(scoring_table=scoring_table, to_score=source_table)

        return scoring_table

    def score_in_dataset(self, to_score: Dataset | Table) -> GoldPxtTorchDataset:
        """Score the novelty of vectorized samples and return a GoldPxtTorchDataset.

        The fitted scoring tool evaluates query vectors and stores the results in
        the PixelTable table at `table_path`. When `chunk` is set, vectors are scored
        in chunks. An optional reducer transforms the vectors before scoring.

        This method calls `score_in_table` to populate or resume the scoring table,
        then wraps the results in a `GoldPxtTorchDataset`. If `drop_table` is True,
        the method removes the table after creating the dataset. The cached dataset
        remains usable after deletion. It contains raw per-vector scores. Use
        `get_novelty_scores_per_idx` on the scoring table to aggregate scores by sample.

        Args:
            to_score: Dataset or Table to score. Each Dataset item must provide a
                1D floating tensor under `vectorized_key`, and collation must
                produce a 2D tensor of shape `(batch_size, feature_dim)`. Missing
                `idx_vector` values default to row positions, and missing `idx`
                values default to vector IDs. A Table must contain the
                `vectorized_key`, `idx`, and `idx_vector` columns. Vector IDs must be
                unique, and multiple vectors may share a sample ID.

        Returns:
            A `GoldPxtTorchDataset` containing novelty scores under `score_key`,
            sample IDs under `idx`, vector IDs under `idx_vector`, and any
            configured retained columns.

        Raises:
            ValueError: If a scorer is unfitted or validation in
                `score_in_table` fails.
        """
        scored_table = self.score_in_table(to_score)

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
        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.label_key is not None:
            minimal_schema[self.label_key] = pxt.String
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        scoring_table = get_valid_table(
            table=old_scoring_table
            if old_scoring_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )

        if self.vectorized_key not in score_from.columns():
            raise ValueError(
                f"Source table does not contain "
                f"the required column {self.vectorized_key}."
            )

        scoring_table_cols = scoring_table.columns()
        if self.score_key not in scoring_table_cols:
            scoring_table.add_column(if_exists="error", **{self.score_key: pxt.Float})

        if (
            self.include_vectorized_in_table
            and self.vectorized_key not in scoring_table_cols
        ):
            idx_vectors = [row["idx_vector"] for row in score_from.sample(1).collect()]
            if not idx_vectors:
                raise ValueError(
                    f"Source table at {self.table_path} is empty, cannot sample vectorized data for schema inference."
                )
            sample = get_sample_row_from_idx(
                score_from,
                idx_key="idx_vector",
                # make sure to take an existing vector
                idx=idx_vectors[0],
                collate_fn=self.collate_fn,
                expected_keys=[self.vectorized_key],
            )

            vectorized_value = sample[self.vectorized_key].squeeze(0)
            scoring_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        vectorized_value.shape, pxt.Float
                    ]
                }
            )

        if scoring_table.count() > 0:
            to_score_indices = set(
                [
                    row["idx_vector"]
                    for row in score_from.select(score_from.idx_vector)
                    .distinct()
                    .collect()
                ]
            )
            already_in_scoring = set(
                [
                    row["idx_vector"]
                    for row in scoring_table.select(scoring_table.idx_vector)
                    .distinct()
                    .collect()
                ]
            )
            still_to_add = to_score_indices.difference(already_in_scoring)
            if not still_to_add:
                logger.info(
                    f"The scoring table is already initialized in {self.table_path}"
                )
                return scoring_table

        col_list = ["idx", self.vectorized_key]
        score_from_cols = score_from.columns()
        if self.to_keep_schema is not None:
            col_list.extend(list(self.to_keep_schema.keys()))
        if self.score_key in score_from_cols:
            col_list.append(self.score_key)
        if self.label_key is not None and self.label_key in score_from_cols:
            col_list.append(self.label_key)

        self._add_rows_to_scoring_table_from_dataset(
            score_from=GoldPxtTorchDataset(
                score_from.select(
                    *[
                        get_expr_from_column_name(score_from, col)
                        for col in dict.fromkeys(col_list + ["idx_vector"])
                    ]
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
            ValueError: If the result table has an incompatible schema.
        """
        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema

        if self.label_key is not None:
            minimal_schema[self.label_key] = pxt.String

        scoring_table = get_valid_table(
            table=old_scoring_table
            if old_scoring_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )

        scoring_table_cols = scoring_table.columns()
        if self.vectorized_key not in scoring_table_cols:
            sample = get_dataset_sample_dict(
                score_from,
                collate_fn=self.collate_fn,
                expected=[self.vectorized_key],
            )

            vectorized_value = (
                sample[self.vectorized_key].squeeze(0).detach().cpu().numpy()
            )
            scoring_table.add_column(
                **{
                    self.vectorized_key: pxt.Array[  # type: ignore[misc]
                        tuple(vectorized_value.shape), pxt.Float
                    ]
                }
            )

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

        # the vectors already present in the scoring table will be excluded
        # from the batches to process to avoid duplicates and speed up processing
        to_exclude_from_batch = {
            int(row["idx_vector"])
            for row in scoring_table.select(scoring_table.idx_vector).collect()
        }

        ready_to_insert: list[dict[str, Any]] = []

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Initializing rows for the scoring table",
            total=len(dataloader) if hasattr(score_from, "__len__") else None,
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

            # Remove the vectors already present in the scoring table to avoid duplicates and speed up processing
            if to_exclude_from_batch:
                batch = filter_batch_from_indices(
                    batch,
                    to_exclude_from_batch,
                    index_key="idx_vector",
                )

                if len(batch) == 0:
                    continue  # all samples already described

            if self.score_key not in batch:
                batch[self.score_key] = [
                    None for _ in range(len(batch[self.vectorized_key]))
                ]

            # add newly added vectors to the exclusion set to avoid duplicates in the next batches
            to_exclude_from_batch.update(
                [
                    int(idx.item() if isinstance(idx, torch.Tensor) else idx)
                    for idx in batch["idx_vector"]
                ]
            )
            to_insert_keys = ["idx", self.score_key]
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
                scoring_table.insert(ready_to_insert)
                ready_to_insert = []

        if ready_to_insert:
            scoring_table.insert(ready_to_insert)

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
        """Sequentially score each label independently, or all vectors without labels.

        Each completed chunk is written to PixelTable by vector ID.

        Args:
            scoring_table: The table in which to store novelty scores.
            to_score: The source table containing query vectors.

        Raises:
            ValueError: If a label has no scoring tool, a pending vector is missing,
                or scoring returns invalid data.
        """
        if self.label_key is not None:
            label_col = get_expr_from_column_name(scoring_table, self.label_key)
            present_labels = set(
                row[self.label_key]
                for row in scoring_table.select(label_col).distinct().collect()
            )
            missing_labels = set(present_labels).difference(set(self.scorer.keys()))
            if missing_labels:
                raise ValueError(f"No scorer for labels {missing_labels}.")
            labels = sorted(present_labels)
        else:
            labels = [None]

        for label_value in labels:
            label_scope = (
                f" for label '{label_value}'" if label_value is not None else ""
            )
            logger.info(f"Scoring vectors{label_scope}")
            self._score_label(
                score_from=to_score,
                scoring_table=scoring_table,
                label_value=label_value,
            )

    def _score_label(
        self,
        score_from: Table,
        scoring_table: Table,
        label_value: str | None = None,
    ) -> None:
        """Score pending vectors for one label, or all vectors without labels.

        Args:
            score_from: Source table containing query vectors.
            scoring_table: Destination table containing vector IDs and scores.
            label_value: Label to score, or None to score all vectors.

        Returns:
            Number of chunks scored for this label.

        Raises:
            ValueError: If a pending vector is missing or scoring returns invalid data.
        """
        score_col = get_expr_from_column_name(scoring_table, self.score_key)

        pending_filter = score_col == None  # noqa: E711

        label_scope = ""
        if label_value is not None:
            assert self.label_key is not None
            label_col = get_expr_from_column_name(scoring_table, self.label_key)
            pending_filter = pending_filter & (label_col == label_value)
            label_scope = f" for label '{label_value}'"

        scoring_pool = scoring_table.where(pending_filter)
        pending_vector_indices = sorted(
            int(row["idx_vector"])
            for row in scoring_pool.select(scoring_table.idx_vector).collect()
        )
        if not pending_vector_indices:
            return

        scorer = (
            self.scorer[label_value]
            if label_value is not None
            else self.scorer["all_labels"]
        )
        vector_table = scoring_table if self.include_vectorized_in_table else score_from
        vector_col = get_expr_from_column_name(vector_table, self.vectorized_key)

        chunk_size = (
            self.chunk if self.chunk is not None else len(pending_vector_indices)
        )
        for chunk_id, start in enumerate(
            range(0, len(pending_vector_indices), chunk_size)
        ):
            chunk_vector_indices = pending_vector_indices[start : start + chunk_size]
            rows = (
                vector_table.where(vector_table.idx_vector.isin(chunk_vector_indices))
                .select(vector_table.idx_vector, vector_col)
                .collect()
            )

            vector_indices = [int(row["idx_vector"]) for row in rows]
            vectors = torch.stack(
                [torch.as_tensor(row[self.vectorized_key]) for row in rows]
            )
            scores = self._compute_scores(vectors, scorer)
            scoring_table.batch_update(
                [
                    {"idx_vector": idx, self.score_key: float(score.item())}
                    for idx, score in zip(vector_indices, scores, strict=True)
                ]
            )
            logger.info(
                f"Scored {len(vector_indices)} vectors{label_scope} from chunk {chunk_id}"
            )

    def _compute_scores(
        self, vectors: torch.Tensor, scorer: GoldNoveltyScoringTool
    ) -> torch.Tensor:
        """Apply the optional reducer and score query vectors in their input order.

        Args:
            vectors: Query vectors shaped `(num_vectors, feature_dim)`.
            scorer: Tool used to score these vectors.

        Returns:
            A tensor of shape `(num_vectors,)` containing one score per query.

        Raises:
            ValueError: If the reducer changes the row count or returns invalid
                vectors, or the scorer fails to return one finite score per query.
        """
        if self.reducer is not None:
            vectors = self.reducer.transform(vectors)

        scores = scorer.score(vectors)
        validate_floating_tensor(scores, 1, len(vectors), "GoldNoveltyScorer")
        return scores

    @staticmethod
    def get_novelty_scores_per_idx(
        table: Table,
        score_key: str = "novelty",
        idx_key: Literal["idx", "idx_vector"] = "idx_vector",
        label_key: str | None = None,
        label_value: str | None = None,
        threshold: float | None = None,
        k: int | None = None,
        best: bool = True,
        aggregation: Callable[[list[float]], float] = max,
    ) -> dict[int, float]:
        """Retrieve computed novelty scores by vector or sample ID.

        Label filtering is applied before aggregating vector scores by sample.
        Threshold filtering and ranking are applied after aggregation.

        Args:
            table: PixelTable table with per-vector novelty scores. Pending rows are
                omitted.
            score_key: Score column name. Defaults to "novelty".
            idx_key: "idx_vector" returns individual vector scores (the default).
                "idx" aggregates available vector scores for each sample.
            label_key: Optional label column to filter. Requires `label_value`.
            label_value: Label to select when `label_key` is provided.
            threshold: Return only scores strictly above this value when
                best=True, or strictly below it when best=False. Defaults to
                no threshold.
            k: Maximum number of scores to return, or None for all matches.
                Must be a nonnegative integer; zero returns no scores.
            best: With `k`, return the highest scores when True (the default),
                or the lowest scores when False. Also controls the threshold
                comparison direction, even when k is None.
            aggregation: Function applied to each sample's list of completed vector
                scores after label filtering. Defaults to max. Ignored for
                "idx_vector".

        Returns:
            A dictionary mapping the requested IDs to their novelty scores.
            Returns an empty dictionary when no completed scores match. When `k`
            is provided, entries are ordered from highest to lowest for `best=True`
            and lowest to highest otherwise; ties retain retrieval order.

        Raises:
            ValueError: If the index key is unsupported or `label_key` is provided
                without `label_value`, or `k` is not a nonnegative integer or None.
        """
        validate_positive_integer(k, "k", allow_none=True, allow_zero=True)

        if idx_key not in ("idx", "idx_vector"):
            raise ValueError("idx_key must be either 'idx' or 'idx_vector'.")

        idx_col = get_expr_from_column_name(table, idx_key)
        score_col = get_expr_from_column_name(table, score_key)

        pool = table.where(score_col != None)  # noqa: E711
        if label_key is not None:
            if label_value is None:
                raise ValueError(
                    "If label_key is provided, label_value must also be provided."
                )
            label_col = get_expr_from_column_name(table, label_key)
            pool = pool.where(label_col == label_value)
        rows = pool.select(idx_col, score_col).collect()

        scores: dict[int, float]
        if idx_key == "idx_vector":
            scores = {int(row[idx_key]): float(row[score_key]) for row in rows}
        else:
            scores_per_index = defaultdict(list)
            for row in rows:
                scores_per_index[int(row[idx_key])].append(float(row[score_key]))
            scores = {
                idx: float(aggregation(values))
                for idx, values in scores_per_index.items()
            }

        if threshold is not None:
            scores = {
                idx: score
                for idx, score in scores.items()
                if (score > threshold if best else score < threshold)
            }

        if k is not None:
            scores = dict(
                sorted(scores.items(), key=lambda item: item[1], reverse=best)[:k]
            )
        return scores
