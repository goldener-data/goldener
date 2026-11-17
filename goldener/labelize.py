"""Predict labels from annotated reference vectors and persist class confidences.

- `GoldLabelPredictionTool` defines the tensor-facing fitted prediction interface.
- `GoldKnnLabelPredictionTool` computes class probabilities from neighbor votes.
- `GoldLabelPredictor` orchestrates reference fitting and resumable, vector-indexed
  prediction for datasets and local PixelTable tables.

Use after `goldener.vectorize.GoldVectorizer`, optionally on reference samples
chosen with `goldener.select.GoldSelector`.
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
from tqdm import tqdm

from goldener.do import GoldDoerWithTable
from goldener.pxt_utils import (
    GoldPxtTorchDataset,
    check_pxt_table_has_primary_key,
    get_expr_from_column_name,
    get_valid_table,
    make_batch_ready_for_table,
)
from goldener.select import DistanceType
from goldener.torch_utils import get_dataset_sample_dict
from goldener.utils import (
    filter_batch_from_indices,
    validate_positive_integer,
    validate_vectorized_input,
)

logger = getLogger(__name__)


class GoldLabelPredictionTool(ABC):
    """Fit annotated vectors and return a confidence for each class.

    Attributes:
        device: Device used for computation, default CPU.
        fitted: Whether the tool is ready to predict.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        """Initialize the computation device.

        Args:
            device: Torch device used for computation.
        """
        self.device = torch.device(device)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate finite floating vectors shaped `(num_vectors, feature_dim)`."""
        validate_vectorized_input(x, type(self).__name__)

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Whether reference vectors and targets have been fitted."""

    @abstractmethod
    def fit(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Fit reference vectors and contiguous integer class IDs.

        Args:
            x: Nonempty floating matrix `(num_vectors, feature_dim)`.
            target: Integer vector `(num_vectors,)` with class IDs `0..num_classes-1`.

        Raises:
            ValueError: If vectors or targets violate this contract.
        """

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities for every query vector and fitted class.

        Args:
            x: Floating matrix `(num_vectors, feature_dim)`.

        Returns:
            Floating `(num_vectors, num_classes)` tensor with columns ordered by
            fitted class ID, on the query device and with its dtype. Rows sum to one.

        Raises:
            ValueError: If unfitted or the input is invalid.
        """


class GoldKnnLabelPredictionTool(GoldLabelPredictionTool):
    """Predict class probabilities as the fraction of nearest neighbors per class.

    The most represented class has the highest probability. Equal distances use
    reference order to break ties. No randomness is used. References are cloned at
    fit time; query blocks bound pairwise distance memory. All references remain
    in memory. When fewer than `k` references exist, all are used.

    Attributes:
        k: Positive number of neighbors, default 5.
        distance: Distance metric, default cosine.
        distance_batch_size: Maximum query rows per distance block, default 1024.
        reference_bank: Fitted vectors on `device`, or None before fitting.
        targets: Cloned reference class IDs, or None before fitting.
        num_classes: Number of contiguous fitted classes.
    """

    def __init__(
        self,
        k: int = 5,
        distance: DistanceType = DistanceType.COSINE,
        device: torch.device | str = "cpu",
        distance_batch_size: int = 1024,
    ) -> None:
        """Configure exact nearest-neighbor voting.

        Args:
            k: Positive neighbor count, capped at the fitted reference count.
            distance: Pairwise distance metric.
            device: Computation device.
            distance_batch_size: Positive query block size.

        Raises:
            ValueError: If a size is not a positive integer.
        """
        super().__init__(device)
        validate_positive_integer(k, "k")
        validate_positive_integer(distance_batch_size, "distance_batch_size")
        self.k = k
        self.distance = distance
        self.distance_batch_size = distance_batch_size
        self.reference_bank: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.num_classes = 0

    @property
    def fitted(self) -> bool:
        """Whether vectors and targets have been fitted."""
        return self.reference_bank is not None

    def fit(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Clone reference vectors and their integer targets.

        Args:
            x: Nonempty finite floating matrix `(num_vectors, feature_dim)`.
            target: Integer class IDs `(num_vectors,)`, covering `0..num_classes-1`.

        Raises:
            ValueError: If references or targets are invalid.
        """
        self._validate_input(x)
        if not len(x):
            raise ValueError("The label reference must not be empty.")
        if (
            not isinstance(target, torch.Tensor)
            or target.shape != (len(x),)
            or target.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError(
                "target must contain one integer class ID per reference vector."
            )
        classes = torch.unique(target).cpu()
        if not torch.equal(classes, torch.arange(len(classes))):
            raise ValueError(
                "target class IDs must cover consecutive integers starting at zero."
            )
        self.reference_bank = x.detach().to(self.device).clone()
        self.targets = target.detach().to(device=self.device, dtype=torch.long).clone()
        self.num_classes = len(classes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class vote fractions for query vectors.

        Args:
            x: Floating matrix `(num_vectors, feature_dim)`.

        Returns:
            Probabilities `(num_vectors, num_classes)` on the query device/dtype.

        Raises:
            ValueError: If unfitted, feature widths differ, or distances are nonfinite.
        """
        self._validate_input(x)
        if self.reference_bank is None or self.targets is None:
            raise ValueError("Fit the label predictor before calling predict.")
        if x.shape[1] != self.reference_bank.shape[1]:
            raise ValueError(
                "Query feature_dim must match the fitted reference feature_dim."
            )
        dtype = torch.promote_types(x.dtype, self.reference_bank.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        reference = self.reference_bank.to(dtype=dtype)
        query = x.detach().to(device=self.device, dtype=dtype)
        result = torch.zeros(
            (len(x), self.num_classes), device=self.device, dtype=dtype
        )
        k = min(self.k, len(reference))
        for start in range(0, len(x), self.distance_batch_size):
            distances = self.distance.distance_function(
                query[start : start + self.distance_batch_size], reference
            )
            if not torch.isfinite(distances).all():
                raise ValueError("Computed neighbor distances must be finite.")
            neighbors = distances.argsort(dim=1, stable=True)[:, :k]
            votes = self.targets[neighbors]
            result[start : start + len(votes)].scatter_add_(
                1, votes, torch.ones_like(votes, dtype=dtype)
            )
        return (result / k).to(device=x.device, dtype=x.dtype)


class GoldLabelPredictor(GoldDoerWithTable):
    """Fit annotated references and persist one class-confidence mapping per vector.

    Dataset items contain one 1D vector; Table inputs require `idx`, unique
    `idx_vector`, and a vector column. Dataset IDs are preserved when present,
    otherwise row positions supply vector IDs and sample IDs default to vector IDs.
    Local PixelTable storage reuses completed predictions on repeated/resumed calls.
    All rows are staged before prediction; `max_batches` limits pending query batches.
    References and fitted tools remain in memory and must be restored after restart.
    Keep inputs, reference class order, and tool settings unchanged across resumes.

    Reference labels are strings or targets mapped with `target_to_label`. Null and
    excluded labels are omitted from fitting. Annotated query rows retain their
    label as a one-hot confidence and must belong to a fitted class; excluded/null
    annotations are predicted normally.
    Stored JSON mappings include every fitted class, making results self-describing.
    Multiple vectors retain their sample's `idx` and distinct `idx_vector`.

    Attributes:
        predictor: Tensor prediction tool; defaults to a new kNN tool.
        vectorized_key: Input vector key, default "vectorized".
        prediction_key: Output confidence mapping key, default "prediction".
        label_key: Annotation key; None uses "label" when fitting or reading annotations.
        target_to_label: Optional mapping from target tuples to class names.
        exclude_labels: Reference/annotation labels to ignore, or None.
        classes: Fitted class names in tensor output column order; `fit` sorts them.
        include_vectorized_in_table: Retain Table input vectors, default False.
            Dataset vectors are always retained. bfloat16 is persisted as float32.
        chunk: Query prediction batch size, or None to use `batch_size`.
        table_path: Local PixelTable destination, keyed by `idx_vector`.
        to_keep_schema: Additional retained metadata columns, or None.
        collate_fn: Dataset collation; defaults to the shared sequence-preserving helper.
        batch_size: Dataset ingestion batch size, default 1.
        num_workers: DataLoader worker count, default 0.
        min_pxt_insert_size: Minimum buffered writes, default 100.
        allow_existing: Reuse the destination table, default True.
        drop_table: Drop the result table after dataset snapshot creation, default False.
        max_batches: Maximum pending prediction batches per call, or None for all.
        distribute: Distributed processing, unsupported and always False.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
        "idx_vector": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        table_path: str,
        predictor: GoldLabelPredictionTool | None = None,
        vectorized_key: str = "vectorized",
        prediction_key: str = "prediction",
        label_key: str | None = None,
        target_to_label: dict[tuple[int, ...], str] | None = None,
        exclude_labels: set[str] | None = None,
        include_vectorized_in_table: bool = False,
        to_keep_schema: dict[str, type] | None = None,
        min_pxt_insert_size: int = 100,
        chunk: int | None = None,
        collate_fn: Callable | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        allow_existing: bool = True,
        distribute: bool = False,
        drop_table: bool = False,
        max_batches: int | None = None,
        classes: list[str] | None = None,
    ) -> None:
        """Configure label prediction and result persistence.

        Args:
            table_path: Result table path.
            predictor: Prediction tool, or None for kNN.
            vectorized_key: Input vector column.
            prediction_key: Output JSON class-probability column.
            label_key: Annotation column; None falls back to "label".
            target_to_label: Optional target tuple to label mapping.
            exclude_labels: Labels ignored during fitting and annotation preservation.
            include_vectorized_in_table: Retain source Table vectors in the destination.
            to_keep_schema: Additional metadata columns to retain.
            min_pxt_insert_size: Positive minimum buffered insertion/update size.
            chunk: Positive prediction batch size, or None for `batch_size`.
            collate_fn: Optional dataset collator.
            batch_size: Positive dataset batch size.
            num_workers: Nonnegative DataLoader worker count.
            allow_existing: Permit resuming existing result tables.
            distribute: Unsupported; must be False.
            drop_table: Drop table after creating a dataset snapshot.
            max_batches: Nonnegative pending prediction batch limit, or None.
            classes: Class names in output column order for an already fitted tool.
                Not needed when fitting through this doer; `fit` replaces them.

        Raises:
            ValueError: If sizes or column names are invalid.
            NotImplementedError: If distributed processing is requested.
        """
        for name, value in (
            ("batch_size", batch_size),
            ("min_pxt_insert_size", min_pxt_insert_size),
        ):
            validate_positive_integer(value, name)
        if chunk is not None:
            validate_positive_integer(chunk, "chunk")
        for name, optional_value in (
            ("num_workers", num_workers),
            ("max_batches", max_batches),
        ):
            if optional_value is not None and (
                isinstance(optional_value, bool)
                or not isinstance(optional_value, int)
                or optional_value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer.")
        if {vectorized_key, prediction_key}.intersection(to_keep_schema or {}):
            raise ValueError(
                "to_keep_schema cannot redefine vector or prediction columns."
            )
        self.predictor = (
            predictor if predictor is not None else GoldKnnLabelPredictionTool()
        )
        self.vectorized_key = vectorized_key
        self.prediction_key = prediction_key
        self.target_to_label = target_to_label
        self.exclude_labels = exclude_labels
        self.include_vectorized_in_table = include_vectorized_in_table
        self.chunk = chunk
        if classes is not None and (
            not classes
            or any(not isinstance(name, str) for name in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("classes must contain distinct string class names.")
        self.classes: list[str] = list(classes) if classes is not None else []
        self.distribute = distribute
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
        return False

    @distribute.setter
    def distribute(self, value: bool) -> None:
        if value:
            raise NotImplementedError(
                "Distributed processing is not implemented for GoldLabelPredictor."
            )

    def _label(self, value: Any) -> str | None:
        """Normalize an annotation, treating null/excluded labels as unannotated."""
        if value is None:
            return None
        if self.target_to_label is not None and not isinstance(value, str):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = self.target_to_label.get(
                tuple(np.asarray(value).reshape(-1).tolist())
            )
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                "Labels must be strings or targets mapped by target_to_label."
            )
        return None if self.exclude_labels and value in self.exclude_labels else value

    def fit(self, reference: Dataset | Table) -> None:
        """Fit the prediction tool on all annotated reference vectors.

        Args:
            reference: Vectorized Dataset or Table with an annotation column.

        Raises:
            ValueError: If no labeled references exist or completed predictions prevent refitting.
        """
        try:
            existing = pxt.get_table(self.table_path, if_not_exists="ignore")
        except Error:
            existing = None
        if (
            existing is not None
            and self.prediction_key in existing.columns()
            and self.predictor.fitted
        ):
            col = get_expr_from_column_name(existing, self.prediction_key)
            if existing.where(col != None).count():  # noqa: E711
                raise ValueError(
                    "Cannot refit after the destination contains completed predictions."
                )
        if isinstance(reference, Table):
            reference = GoldPxtTorchDataset(reference, keep_cache=False)
        vectors: list[torch.Tensor] = []
        labels: list[str] = []
        key = self.label_key or "label"
        for batch in DataLoader(
            reference,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        ):
            if key not in batch or self.vectorized_key not in batch:
                raise ValueError(f"Reference requires {key} and {self.vectorized_key}.")
            self.predictor._validate_input(batch[self.vectorized_key])
            for vector, value in zip(batch[self.vectorized_key], batch[key]):
                label = self._label(value)
                if label is not None:
                    vectors.append(vector)
                    labels.append(label)
        if not vectors:
            raise ValueError("The label reference must contain annotated vectors.")
        classes = sorted(set(labels))
        targets = torch.tensor(
            [classes.index(label) for label in labels], dtype=torch.long
        )
        self.predictor.fit(torch.stack(vectors), targets)
        self.classes = classes

    def fit_predictor(self, reference: Dataset | Table) -> None:
        """Fit the predictor; alias for `fit`.

        Args:
            reference: Annotated vectorized Dataset or Table.
        """
        self.fit(reference)

    def predict_in_dataset(
        self, to_label: Dataset | Table, reference: Dataset | Table | None = None
    ) -> GoldPxtTorchDataset:
        """Predict via `predict_in_table` and return a snapshot-backed dataset.

        Args:
            to_label: Vectorized queries.
            reference: Optional annotated reference used to fit before prediction.

        Returns:
            Dataset containing vector IDs, sample IDs, and class-confidence mappings.
            Pending predictions remain null when `max_batches` limits the call.
        """
        table = self.predict_in_table(to_label, reference)
        dataset = GoldPxtTorchDataset(table, keep_cache=True)
        if self.drop_table:
            pxt.drop_table(table)
        return dataset

    def predict_in_table(
        self, to_label: Dataset | Table, reference: Dataset | Table | None = None
    ) -> Table:
        """Predict pending query rows and preserve completed results.

        Args:
            to_label: Dataset with 1D vectors or Table with vectors and IDs.
            reference: Optional reference for fitting. Required before the first prediction.

        Returns:
            Table with a nullable JSON class-confidence mapping per vector.

        Raises:
            ValueError: If unfitted, class names are unavailable, or inputs are invalid.
        """
        if reference is not None:
            self.fit(reference)
        if not self.predictor.fitted or not self.classes:
            raise ValueError("Fit with a reference dataset/table before predicting.")
        table = self.setup_prediction_table(to_label)
        self._sequential_predict(
            table, to_label if isinstance(to_label, Table) else table
        )
        return table

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

    def setup_prediction_table(self, to_label: Dataset | Table) -> Table:
        """Set up the prediction table based on the source dataset or table.

        Args:
            to_label: Dataset or Table containing already vectorized samples.
                Missing dataset indices are generated from row positions.

        Returns:
            A PixelTable Table populated with source indices and retained columns.
            Existing predictions are preserved; new rows have null predictions.

        Raises:
            ValueError: If `allow_existing` is False and the table already exists,
                or required columns, vectors, indices, or stored schema are invalid.
        """
        self._query_labels: dict[int, str] = {}
        logger.info(f"Loading the existing prediction table from {self.table_path}")
        try:
            old_prediction_table = pxt.get_table(
                self.table_path, if_not_exists="ignore"
            )
        except Error:
            logger.info(f"No existing prediction table from {self.table_path}")
            old_prediction_table = None

        if old_prediction_table is not None:
            if not self.allow_existing:
                raise ValueError(
                    f"Table at path {self.table_path} already exists and allow_existing is set to False."
                )
            check_pxt_table_has_primary_key(old_prediction_table, {"idx_vector"})
            expected_schema = self._MINIMAL_SCHEMA | (self.to_keep_schema or {})
            if self.prediction_key in old_prediction_table.columns():
                expected_schema[self.prediction_key] = pxt.Json
            for key, schema_type in expected_schema.items():
                if key not in old_prediction_table.columns():
                    raise ValueError(f"Existing result table is missing column {key}.")
                expected = ColumnType.normalize_type(schema_type)
                actual = get_expr_from_column_name(old_prediction_table, key).col_type
                if not expected.matches(actual):
                    raise ValueError(f"Existing column {key} has an incompatible type.")

        if isinstance(to_label, Table):
            prediction_table = self._prediction_table_from_table(
                to_label, old_prediction_table
            )
        else:
            prediction_table = self._prediction_table_from_dataset(
                to_label, old_prediction_table
            )
        return prediction_table

    def _prediction_table_from_table(
        self, predict_from: Table, old_prediction_table: Table | None
    ) -> Table:
        """Create or validate the prediction table schema from a PixelTable table."""
        required_columns = {"idx", "idx_vector", self.vectorized_key}
        required_columns.update(self.to_keep_schema or {})
        if missing := required_columns.difference(predict_from.columns()):
            raise ValueError(f"Source table is missing required columns: {missing}.")

        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema
        prediction_table = get_valid_table(
            table=old_prediction_table
            if old_prediction_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )
        if self.prediction_key not in prediction_table.columns():
            prediction_table.add_column(
                if_exists="error", **{self.prediction_key: pxt.Json}
            )
        if self.include_vectorized_in_table:
            vector_type = get_expr_from_column_name(
                predict_from, self.vectorized_key
            ).col_type
            if self.vectorized_key not in prediction_table.columns():
                prediction_table.add_column(
                    if_exists="error",
                    **{self.vectorized_key: vector_type},  # type: ignore[arg-type]
                )
            elif not vector_type.matches(
                get_expr_from_column_name(
                    prediction_table, self.vectorized_key
                ).col_type
            ):
                raise ValueError("Existing vector column has an incompatible shape.")

        col_list = ["idx", "idx_vector", self.vectorized_key]
        if self.to_keep_schema is not None:
            col_list.extend(self.to_keep_schema)
        annotation_key = self.label_key or "label"
        if annotation_key in predict_from.columns() and annotation_key not in col_list:
            col_list.append(annotation_key)
        self._add_rows_to_prediction_table_from_dataset(
            predict_from=GoldPxtTorchDataset(
                predict_from.select(
                    *[get_expr_from_column_name(predict_from, col) for col in col_list]
                ),
                keep_cache=False,
            ),
            prediction_table=prediction_table,
            include_vectorized=self.include_vectorized_in_table,
        )
        return prediction_table

    def _prediction_table_from_dataset(
        self, predict_from: Dataset, old_prediction_table: Table | None
    ) -> Table:
        """Create or validate the prediction table schema from a PyTorch Dataset."""
        if hasattr(predict_from, "__len__") and len(predict_from) == 0:
            raise ValueError("Vector source must not be empty.")
        sample = get_dataset_sample_dict(
            predict_from, collate_fn=self.collate_fn, expected=[self.vectorized_key]
        )
        self.predictor._validate_input(sample[self.vectorized_key])
        vectorized_value = self._vector_numpy(sample[self.vectorized_key][0])
        vector_type: type = pxt.Array[
            tuple(vectorized_value.shape), vectorized_value.dtype.type
        ]  # type: ignore[misc]

        minimal_schema = self._MINIMAL_SCHEMA.copy()
        if self.to_keep_schema is not None:
            minimal_schema |= self.to_keep_schema
        prediction_table = get_valid_table(
            table=old_prediction_table
            if old_prediction_table is not None
            else self.table_path,
            minimal_schema=minimal_schema,
            primary_key="idx_vector",
        )
        if self.vectorized_key not in prediction_table.columns():
            prediction_table.add_column(
                if_exists="error", **{self.vectorized_key: vector_type}
            )
        elif not ColumnType.normalize_type(vector_type).matches(
            get_expr_from_column_name(prediction_table, self.vectorized_key).col_type
        ):
            raise ValueError("Existing vector column has an incompatible shape.")
        if self.prediction_key not in prediction_table.columns():
            prediction_table.add_column(
                if_exists="error", **{self.prediction_key: pxt.Json}
            )

        self._add_rows_to_prediction_table_from_dataset(
            predict_from, prediction_table, include_vectorized=True
        )
        return prediction_table

    def _add_rows_to_prediction_table_from_dataset(
        self,
        predict_from: Dataset,
        prediction_table: Table,
        include_vectorized: bool = False,
    ) -> None:
        """Populate the prediction table in batches, preserving completed rows.

        Missing vector IDs default to row positions; missing sample IDs default to
        vector IDs. All rows are ingested before prediction, independently of `max_batches`.
        """
        dataloader = DataLoader(
            predict_from,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        to_exclude_from_batch = {
            int(row["idx_vector"])
            for row in prediction_table.select(prediction_table.idx_vector).collect()
        }
        seen_indices: set[int] = set()
        feature_dim = None
        ready_to_insert: list[dict[str, Any]] = []
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Initializing rows for the prediction table",
            total=len(dataloader) if hasattr(predict_from, "__len__") else None,
        ):
            if self.vectorized_key not in batch:
                raise ValueError(
                    f"Dataset does not contain required key {self.vectorized_key}."
                )
            vectors = batch[self.vectorized_key]
            self.predictor._validate_input(vectors)
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
                batch[key] = [self._normalize_index(value) for value in batch[key]]
            indices = batch["idx_vector"]
            if len(set(indices)) != len(indices) or seen_indices.intersection(indices):
                raise ValueError("Source idx_vector values must be unique.")
            seen_indices.update(indices)
            annotation_key = self.label_key or "label"
            if annotation_key in batch:
                for idx, value in zip(indices, batch[annotation_key]):
                    label = self._label(value)
                    if label is not None:
                        if label not in self.classes:
                            raise ValueError(
                                f"Query annotation {label!r} is absent from the fitted reference classes."
                            )
                        self._query_labels[idx] = label

            if to_exclude_from_batch:
                batch = filter_batch_from_indices(
                    batch, to_exclude_from_batch, index_key="idx_vector"
                )
                if not batch:
                    continue

            batch[self.prediction_key] = [None] * len(batch["idx_vector"])
            to_insert_keys = ["idx", self.prediction_key]
            if self.to_keep_schema is not None:
                to_insert_keys.extend(self.to_keep_schema)
                for key in self.to_keep_schema:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].detach().cpu().numpy()
            if include_vectorized:
                to_insert_keys.append(self.vectorized_key)
                batch[self.vectorized_key] = self._vector_numpy(
                    batch[self.vectorized_key]
                )
            ready_to_insert.extend(
                make_batch_ready_for_table(batch, to_insert_keys, "idx_vector")
            )
            if len(ready_to_insert) >= self.min_pxt_insert_size:
                prediction_table.insert(ready_to_insert)
                ready_to_insert = []
        if ready_to_insert:
            prediction_table.insert(ready_to_insert)
        if not seen_indices:
            raise ValueError("Vector source must not be empty.")

    def _sequential_predict(self, table: Table, to_label: Table) -> None:
        """Predict pending query chunks against the same fitted reference."""
        prediction_col = get_expr_from_column_name(table, self.prediction_key)
        pending = [
            int(row["idx_vector"])
            for row in table.where(prediction_col == None)  # noqa: E711
            .select(table.idx_vector)
            .collect()
        ]
        vector_table = table if self.include_vectorized_in_table else to_label
        vector_col = get_expr_from_column_name(vector_table, self.vectorized_key)
        batch_size = self.chunk or self.batch_size
        updates: list[dict[str, Any]] = []
        for batch_idx, start in enumerate(range(0, len(pending), batch_size)):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            ids = pending[start : start + batch_size]
            rows = (
                vector_table.where(vector_table.idx_vector.isin(ids))
                .select(vector_table.idx_vector, vector_col)
                .collect()
            )
            if len(rows) != len(ids):
                raise ValueError(
                    "A pending destination vector is missing from the supplied source."
                )
            vectors = torch.stack(
                [torch.as_tensor(row[self.vectorized_key]) for row in rows]
            )
            predictions = self._compute_predictions(vectors)
            for i, row in enumerate(rows):
                label = self._query_labels.get(int(row["idx_vector"]))
                if label is not None:
                    predictions[i] = {
                        name: float(name == label) for name in self.classes
                    }
            updates.extend(
                {
                    "idx_vector": int(row["idx_vector"]),
                    self.prediction_key: prediction,
                }
                for row, prediction in zip(rows, predictions)
            )
            if len(updates) >= self.min_pxt_insert_size:
                table.batch_update(updates)
                updates = []
            logger.info(f"Predicted {len(rows)} vectors from batch {batch_idx}")
        if updates:
            table.batch_update(updates)

    def _compute_predictions(self, vectors: torch.Tensor) -> list[dict[str, float]]:
        """Validate the tool contract and attach fitted class names."""
        predictions = self.predictor.predict(vectors)
        if (
            not isinstance(predictions, torch.Tensor)
            or predictions.shape != (len(vectors), len(self.classes))
            or not predictions.is_floating_point()
            or not torch.isfinite(predictions).all()
            or (predictions < 0).any()
            or (predictions > 1).any()
            or not torch.allclose(
                predictions.sum(dim=1),
                predictions.new_ones(len(vectors)),
                atol=1e-3,
                rtol=1e-3,
            )
        ):
            raise ValueError(
                "Predictor output must contain one probability per vector and class, with rows summing to one."
            )
        return [
            dict(zip(self.classes, row)) for row in predictions.detach().cpu().tolist()
        ]

    def estimate_computation_time(self, dataset: Dataset) -> float | None:
        """Return None because arbitrary fitted tools cannot be timed state-safely.

        Args:
            dataset: Dataset whose prediction time would be estimated.

        Returns:
            None; prediction is not performed or advanced.
        """
        return None

    @staticmethod
    def get_predictions(
        table: Table, prediction_key: str = "prediction"
    ) -> dict[int, dict[str, float]]:
        """Retrieve completed confidence mappings keyed by vector ID.

        Args:
            table: Prediction result table.
            prediction_key: Confidence mapping column.

        Returns:
            Mapping from `idx_vector` to class probabilities; null rows are omitted.
        """
        return {
            int(row["idx_vector"]): row[prediction_key]
            for row in table.select(
                table.idx_vector, get_expr_from_column_name(table, prediction_key)
            ).collect()
            if row[prediction_key] is not None
        }

    @staticmethod
    def get_confident_indices(
        table: Table,
        min_confidence: float,
        prediction_key: str = "prediction",
        label: str | None = None,
        idx_key: str = "idx",
    ) -> set[tuple[int, str]]:
        """Find confident sample/class pairs using the maximum over their vectors.

        Args:
            table: Prediction result table.
            min_confidence: Inclusive confidence threshold in `[0, 1]`.
            prediction_key: JSON class-confidence mapping column.
            label: Optional class to query; None considers every class.
            idx_key: Sample `idx` or vector `idx_vector`.

        Returns:
            Distinct `(ID, class_name)` pairs with at least one qualifying vector.
            Null predictions are omitted. Several classes can qualify for one ID.

        Raises:
            ValueError: If the threshold or index key is invalid.
        """
        if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be finite and between 0 and 1.")
        if idx_key not in ("idx", "idx_vector"):
            raise ValueError("idx_key must be either 'idx' or 'idx_vector'.")
        result = set()
        for row in table.select(
            get_expr_from_column_name(table, idx_key),
            get_expr_from_column_name(table, prediction_key),
        ).collect():
            for name, confidence in (row[prediction_key] or {}).items():
                if (label is None or name == label) and confidence >= min_confidence:
                    result.add((int(row[idx_key]), name))
        return result
