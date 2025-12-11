from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import Callable

import pixeltable as pxt
import torch
from pixeltable.catalog import Table
from torch.utils.data import Dataset, DataLoader

from goldener.describe import GoldDescriptor
from goldener.pxt_utils import (
    get_expr_from_column_name,
    set_value_to_idx_rows,
    GoldPxtTorchDataset,
)
from goldener.select import GoldSelector
from goldener.vectorize import GoldVectorizer

logger = getLogger(__name__)


@dataclass
class GoldSet:
    """Configuration for a gold set used for splitting.

    Attributes:
        name: Name of the gold set (None if the set is for not selected data).
        ratio: Ratio of samples to assign to this set (between 0 and 1). This value
        cannot be one of 0 or 1 (goal is to select a subset of the full dataset).
    """

    name: str | None
    ratio: float

    def __post_init__(self) -> None:
        """Validate the GoldSet configuration after initialization.

        Raises:
            ValueError: If ratio is not between 0 and 1 (exclusive).
        """
        if not (0 < self.ratio < 1):
            raise ValueError("Ratio must be between 0 and 1.")


class GoldSplitter:
    """Split a dataset into multiple sets based on features.

    The GoldSplitter leverages a GoldDescriptor to extract features from the dataset,
    a GoldVectorizer to vectorize these features, and a GoldSelector to select samples
    for each set based on specified ratios.

    The splitting can operate in a sequential (single-process) mode or a
    distributed mode (not implemented).

    See GoldDescriptor, GoldVectorizer, and GoldSelector for more details on each component.

    Attributes:
        sets: List of GoldSet configurations defining the splits.
        descriptor: GoldDescriptor used to describe the dataset.
        vectorizer: GoldVectorizer used to vectorize the described dataset.
        selector: GoldSelector used to select samples for each set. The collate_fn of the selector
        will be set to `pxt_torch_dataset_collate_fn`, and the select_key will be forced to "features"
        to match the descriptor's output column.
        in_described_table: Whether to return the splitting in the described table or the selected table.
        allow_existing: Whether to allow existing tables in all components.
        drop_table: Whether to drop the described table after splitting.
        max_batches: Optional maximum number of batches to process in both descriptor and selector. Useful for testing on a small subset of the dataset.
    """

    def __init__(
        self,
        sets: list[GoldSet],
        descriptor: GoldDescriptor,
        vectorizer: GoldVectorizer,
        selector: GoldSelector,
        in_described_table: bool = False,
        allow_existing: bool = True,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        """Initialize the GoldSplitter.

        Args:
            sets: List of GoldSet configurations defining the splits.
            descriptor: GoldDescriptor for extracting features from the dataset.
            vectorizer: GoldVectorizer for vectorizing described features.
            selector: GoldSelector for selecting samples for each set.
            in_described_table: Whether to return splits in the described table. Defaults to False.
            allow_existing: Whether to allow existing tables. Defaults to True.
            drop_table: Whether to drop intermediate tables. Defaults to False.
            max_batches: Optional maximum number of batches to process.
        """
        self.sets = sets
        self.descriptor = descriptor
        self.vectorizer = vectorizer
        self.selector = selector
        self.in_described_table = in_described_table
        self.drop_table = drop_table
        self._max_batches = max_batches
        self.allow_existing = allow_existing
        self.max_batches = max_batches

    @property
    def max_batches(self) -> int | None:
        """Get the maximum number of batches to process."""
        return self._max_batches

    @max_batches.setter
    def max_batches(self, value: int | None) -> None:
        """Set the maximum number of batches to process in both descriptor and selector."""
        self._max_batches = value
        self.descriptor.max_batches = value

    @property
    def allow_existing(self) -> bool:
        """Get whether existing tables are allowed in all components."""
        return self._allow_existing

    @allow_existing.setter
    def allow_existing(self, value: bool) -> None:
        """Set whether existing tables are allowed in all components."""
        self._allow_existing = value
        self.descriptor.allow_existing = value
        self.vectorizer.allow_existing = value
        self.selector.allow_existing = value

    def _check_sets_validity(self, sets: list[GoldSet], ratios_sum: float) -> None:
        """Validate the sets configuration.

        This private method ensures that the sum of ratios is valid and that set names are unique.

        Args:
            sets: List of GoldSet configurations to validate.
            ratios_sum: Sum of all set ratios.

        Raises:
            ValueError: If ratios_sum is not between 0 and 1, or if set names are not unique.
        """
        if not (0 < ratios_sum <= 1.0):
            raise ValueError(
                "Sum of split ratios must be greater than 0.0 and at most 1.0"
            )

        set_names = [s.name for s in sets]
        if len(set_names) != len(set(set_names)):
            raise ValueError(f"Set names must be unique, got {set_names}")

    @property
    def sets(self) -> list[GoldSet]:
        """Get the current sets configuration for the splitter."""
        return self._sets

    @sets.setter
    def sets(self, sets: list[GoldSet]) -> None:
        """Set the sets configuration for the splitter.

        Args:
            sets: New list of GoldSet configurations defining the splits.
        """
        ratios_sum = sum([s.ratio for s in sets])

        self._check_sets_validity(sets, ratios_sum)

        self._sets = sets
        if ratios_sum < 1.0:
            self._sets.append(
                GoldSet(
                    name=None,
                    ratio=1.0 - ratios_sum,
                )
            )

    @staticmethod
    def get_split_indices(
        split_data: Table | Dataset,
        selection_key: str,
        idx_key: str = "idx",
        batch_size: int = 1024,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
    ) -> dict[str, set[int]]:
        """Get the indices of samples for each set from a split table.

        Args:
            split_table: PixelTable Table containing the split information.
            selection_key: Column name in the table indicating the set assignment.
            idx_key: Column name in the table indicating the sample indices.
            batch_size: Batch size to use when processing the input as dataset.
            num_workers: Number of workers to use when processing the input as dataset.
            collate_fn: Collate function to use when processing the input as dataset. It should
                return a dictionary with at least the keys specified by `idx_key` and `selection_key`.

        Returns:
            A dictionary mapping each set name to a set of sample indices.
        """

        set_indices: dict[str, set[int]] = defaultdict(set)

        if isinstance(split_data, Dataset):
            dataloader = DataLoader(
                split_data,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

            for batch in dataloader:
                batch_indices = batch[idx_key]
                batch_selections = batch[selection_key]
                for idx_value, set_name in zip(batch_indices, batch_selections):
                    if set_name is None:
                        continue
                    set_indices[set_name].add(
                        int(idx_value.item())
                        if isinstance(idx_value, torch.Tensor)
                        else idx_value
                    )
        else:
            selection_col = get_expr_from_column_name(split_data, selection_key)
            for row in split_data.select(selection_col).distinct().collect():
                set_name = row[selection_key]
                if set_name is None:
                    continue

                set_indices[set_name] = GoldSelector.get_selected_sample_indices(
                    split_data,
                    selection_key=selection_key,
                    value=set_name,
                    idx_key=idx_key,
                )

        return set_indices

    def split_in_dataset(
        self,
        to_split: Dataset | Table,
    ) -> GoldPxtTorchDataset:
        """Split the dataset into multiple sets in a PixelTable table.

        The dataset is first described using the gold descriptor (extracts features), and then samples are selected
        for each set based on the specified ratios after vectorization.

        This method is idempotent (i.e. failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will not duplicate or recompute the splitting decisions
        already present in the PixelTable table.

        Args:
            to_split: Dataset or Table to be split. If a Dataset is provided, each item should be a
            dictionary with at least the key specified by descriptor `data_key` after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
            it should contain both 'idx' and `data_key` column for the descriptor.

        Returns:
            A GoldPxtTorchDataset dataset containing at least the set assignation in the `selection_key` column. Then
            the other columns are either from the described table (if `in_described_table` is True)
            or from the selected table (if `in_described_table` is False).
        """

        split_table = self.split_in_table(to_split)

        split_dataset = GoldPxtTorchDataset(split_table, keep_cache=True)

        if self.drop_table:
            self._drop_tables()

        return split_dataset

    def split_in_table(self, to_split: Dataset | Table) -> Table:
        """Split the dataset into multiple sets in a PixelTable table.

        The dataset is first described using the gold descriptor (extracts features), and then samples are selected
        for each set based on the specified ratios after vectorization.

        This method is idempotent (i.e. failure proof), meaning that if it is called
        multiple times on the same dataset or table, it will not duplicate or recompute the splitting decisions
        already present in the PixelTable table.

        See describe_in_table, vectorize_in_table, and select_in_table for more details on each step.

        Args:
            to_split: Dataset or Table to be split. If a Dataset is provided, each item should be a
            dictionary with at least the key specified by descriptor `data_key` after applying the collate_fn.
            If the collate_fn is None, the dataset is expected to directly provide such batches. If a Table is provided,
            it should contain both 'idx' and `data_key` column for the descriptor.

        Returns:
            A PixelTable Table containing at least the set assignation in the `selection_key` column. Then
            the other columns are either from the described table (if `in_described_table` is True)
            or from the selected table (if `in_described_table` is False).

        Raises:
            ValueError: If any set results in zero samples due to its ratio, if class_key is not found,
            or if class stratification results in zero samples for any class in a set.
        """
        description_table = self.descriptor.describe_in_table(to_split)
        sample_count = description_table.count()

        vectorized_table = self.vectorizer.vectorize_in_table(description_table)

        # select data for all sets except the last one
        for idx_set, gold_set in enumerate(self._sets[:-1]):
            set_count = int(gold_set.ratio * sample_count)
            if set_count == 0:
                raise ValueError(
                    f"Set '{gold_set.name}' has ratio {gold_set.ratio} which results "
                    f"in zero samples for dataset of size {sample_count}."
                )
            selected_table = self.selector.select_in_table(
                vectorized_table, set_count, value=gold_set.name
            )

        # remaining samples are assigned to the last set
        already_selected = self.selector.get_selected_sample_indices(
            selected_table,
            selection_key=self.selector.selection_key,
            value=self._sets[-1].name,
        )
        remaining_idx_list = self.selector.get_selected_sample_indices(
            selected_table,
            selection_key=self.selector.selection_key,
            value=None,
        )
        if len(remaining_idx_list) == 0 and len(already_selected) == 0:
            raise ValueError(
                f"Set '{self._sets[-1].name}' has ratio {self._sets[-1].ratio} which results "
                f"in zero samples for dataset of size {len(remaining_idx_list)}."
            )
        selection_col = get_expr_from_column_name(
            selected_table,
            self.selector.selection_key,
        )
        set_value_to_idx_rows(
            table=selected_table,
            col_expr=selection_col,
            idx_expr=selected_table.idx_sample,
            indices=remaining_idx_list,
            value=self._sets[-1].name,
        )
        split_table = selected_table
        if self.in_described_table:
            for set_cfg in self._sets:
                set_name = set_cfg.name
                set_indices = self.selector.get_selected_sample_indices(
                    selected_table,
                    selection_key=self.selector.selection_key,
                    value=set_name,
                )
                if self.selector.selection_key not in description_table.columns():
                    description_table.add_column(
                        if_exists="error",
                        **{self.selector.selection_key: pxt.String},
                    )
                set_value_to_idx_rows(
                    description_table,
                    col_expr=get_expr_from_column_name(
                        description_table, self.selector.selection_key
                    ),
                    idx_expr=description_table.idx,
                    indices=set_indices,
                    value=set_name,
                )
            split_table = description_table

        if self.drop_table:
            to_drop = []
            if self.vectorizer.table_path != self.selector.table_path:
                to_drop.append(self.vectorizer.table_path)

            if self.in_described_table:
                to_drop.append(self.selector.table_path)
            else:
                to_drop.append(self.descriptor.table_path)

            for table_name in to_drop:
                pxt.drop_table(table_name, if_not_exists="ignore")

        return split_table

    def _drop_tables(self) -> None:
        """Drop all intermediate tables created during the splitting process.

        This private method cleans up the descriptor, vectorizer, and selector tables
        if drop_table is enabled.
        """
        if self.drop_table:
            for table_name in (
                self.descriptor.table_path,
                self.vectorizer.table_path,
                self.selector.table_path,
            ):
                pxt.drop_table(table_name, if_not_exists="ignore")
