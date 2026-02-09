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
from goldener.utils import (
    check_sampling_size,
    check_all_same_type,
    get_sampling_count_from_size,
)
from goldener.vectorize import GoldVectorizer


logger = getLogger(__name__)


@dataclass
class GoldSet:
    """Configuration for a gold set used for splitting.

    Attributes:
        name: Name of the gold set (None if the set is for not selected data).
        size: Size (ratio or actual number) of samples to assign to this set (between 0 and 1). If a ratio, this value
        cannot be one of 0 or 1 (goal is to select a subset of the full dataset). If a number, it must be strictly
        positive and less than the total number of samples in the dataset.
    """

    name: str | None
    size: float | int

    def __post_init__(self) -> None:
        """Validate the GoldSet configuration after initialization.

        Raises:
            ValueError: If ratio is not between 0 and 1 (exclusive).
        """
        if isinstance(self.size, float) and not (0 < self.size < 1):
            raise ValueError("Size as float must be between 0 and 1.")
        elif isinstance(self.size, int) and self.size <= 0:
            raise ValueError("Size as int must be strictly positive.")


def check_sets_validity(
    sets: list[GoldSet], total: int | None = None, force_max: bool = True
) -> None:
    """Validate the sets configuration.

    This function ensures that the sum of ratios is valid and that set names are unique.

    Args:
        sets: List of GoldSet configurations to validate.
        total: Optional total number of samples (used if sizes are integers).
        force_max: Whether to enforce that the sum of ratios equals 1.0 (for float sizes) or total (for int sizes).

    Raises:
        ValueError: If the set's sizes are not valid, or if set names are not unique.
    """
    sizes = [s.size for s in sets]
    check_all_same_type(sizes)
    check_sampling_size(sum(sizes), total_size=total, force_max=force_max)

    set_names = [s.name for s in sets]
    if len(set_names) != len(set(set_names)):
        raise ValueError(f"Set names must be unique, got {set_names}")


class GoldSplitter:
    """Split a dataset into multiple sets based on features.

    The GoldSplitter leverages a GoldDescriptor to extract features from the dataset,
    a GoldVectorizer to vectorize these features, and a GoldSelector to select samples
    for each set based on specified ratios.

    The splitting can operate in a sequential (single-process) mode or a
    distributed mode (not implemented).

    At least 2 sets are required to split data and every sample
    will be associated to a unique set using a GoldSelector. The last set is filled with the remaining elements.
    Every set is at least associated with 1 points (ratio might not be fully matched for small data regime).

    See GoldDescriptor, GoldVectorizer, and GoldSelector for more details on each component.

    Attributes:
        sets: List of GoldSet configurations defining the splits.
        selector: GoldSelector used to select samples for each set. The collate_fn of the selector
        will be set to `pxt_torch_dataset_collate_fn`, and the select_key will be forced to "features"
        to match the descriptor's output column.
        descriptor: Optional GoldDescriptor used to describe the dataset.
        vectorizer: Optional GoldVectorizer used to vectorize the described dataset.
        in_described_table: Whether to return the splitting in the described table or the selected table.
        allow_existing: Whether to allow existing tables in all components.
        drop_table: Whether to drop intermediate tables. Defaults to False.
        max_batches: Optional maximum number of batches to process in both descriptor and selector. Useful for testing on a small subset of the dataset.
    """

    def __init__(
        self,
        sets: list[GoldSet],
        selector: GoldSelector,
        descriptor: GoldDescriptor | None = None,
        vectorizer: GoldVectorizer | None = None,
        in_described_table: bool = False,
        allow_existing: bool = True,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        """Initialize the GoldSplitter.

        Args:
            sets: List of GoldSet configurations defining the splits.
            selector: GoldSelector for selecting samples for each set.
            descriptor: Optional GoldDescriptor for extracting features from the dataset.
            vectorizer: Optional GoldVectorizer for vectorizing described features.
            in_described_table: Whether to return splits in the described table. Defaults to False.
            allow_existing: Whether to allow existing tables. Defaults to True.
            drop_table: Whether to drop intermediate tables. Defaults to False.
            max_batches: Optional maximum number of batches to process.
        """
        if descriptor is None and in_described_table:
            raise ValueError(
                "in_described_table is set to True, but no descriptor is provided."
            )

        if vectorizer is not None:
            if vectorizer.vectorized_key != selector.vectorized_key:
                raise ValueError(
                    f"Vectorizer vectorized_key '{vectorizer.vectorized_key}' does not match "
                    f"selector's vectorized_key '{selector.vectorized_key}'. They must be the same."
                )

            if (
                descriptor is not None
                and descriptor.description_key != vectorizer.data_key
            ):
                raise ValueError(
                    f"Descriptor description_key '{descriptor.description_key}' does not match "
                    f"vectorizer's data_key '{vectorizer.data_key}'. They must be the same."
                )

        elif (
            descriptor is not None
            and descriptor.description_key != selector.vectorized_key
        ):
            raise ValueError(
                f"Descriptor description_key '{descriptor.description_key}' does not match "
                f"selector's vectorized_key '{selector.vectorized_key}'. They must be the same."
            )

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
        """Set the maximum number of batches to process during splitting."""
        self._max_batches = value
        if self.descriptor is not None:
            self.descriptor.max_batches = value
        else:
            if self.vectorizer is not None:
                self.vectorizer.max_batches = value
            else:
                self.selector.max_batches = value

    @property
    def allow_existing(self) -> bool:
        """Get whether existing tables are allowed in all components."""
        return self._allow_existing

    @allow_existing.setter
    def allow_existing(self, value: bool) -> None:
        """Set whether existing tables are allowed in all components."""
        self._allow_existing = value
        if self.descriptor is not None:
            self.descriptor.allow_existing = value
        if self.vectorizer is not None:
            self.vectorizer.allow_existing = value
        self.selector.allow_existing = value

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
        if len(sets) < 2:
            raise ValueError("Splitting data requires at least two sets.")

        size_types = [type(s.size) for s in sets]

        if len(set(size_types)) > 1:
            raise ValueError(
                f"All set sizes must be of the same type, got types {size_types}."
            )

        check_sets_validity(sets, force_max=True if size_types[0] is float else False)
        self._sets = sets

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

        At least 2 sets are required to split data and every sample
        will be associated to a unique set using a GoldSelector. The last set is filled with the remaining elements.
        Every set is at least associated with 1 points (ratio might not be fully matched for small data regime).

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

        self._drop_tables(drop_all=True)

        return split_dataset

    def split_in_table(self, to_split: Dataset | Table) -> Table:
        """Split the dataset into multiple sets in a PixelTable table.

        The dataset is first described using the gold descriptor (extracts features), and then samples are selected
        for each set based on the specified ratios after vectorization.

        At least 2 sets are required to split data and every sample
        will be associated to a unique set using a GoldSelector. The last set is filled with the remaining elements.
        Every set is at least associated with 1 points (ratio might not be fully matched for small data regime).

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
        description = (
            self.descriptor.describe_in_table(to_split)
            if self.descriptor is not None
            else to_split
        )

        vectorized = (
            self.vectorizer.vectorize_in_table(description)
            if self.vectorizer is not None
            else description
        )

        if isinstance(vectorized, Table):
            sample_count = vectorized.select(vectorized.idx).distinct().count()
        elif hasattr(vectorized, "__len__"):
            sample_count = len(vectorized)
        else:
            sample_count = None

        if sample_count is not None:
            check_sets_validity(self._sets, total=sample_count, force_max=True)

        # select data for all sets except the last one
        for idx_set, gold_set in enumerate(self._sets):
            logger.info(
                f"Selecting samples for set '{gold_set.name}' with size {gold_set.size}."
            )
            set_count = get_sampling_count_from_size(
                sampling_size=gold_set.size, total_size=sample_count
            )

            if idx_set < len(self._sets) - 1:
                selected_table = self.selector.select_in_table(
                    vectorized, set_count, value=gold_set.name
                )
                if sample_count is None:
                    sample_count = (
                        selected_table.select(selected_table.idx).distinct().count()
                    )
                    check_sets_validity(self._sets, total=sample_count, force_max=True)
            else:
                # The last set is gathering all the remaining samples
                selection_col = get_expr_from_column_name(
                    selected_table, self.selector.selection_key
                )
                already_in_set_count = selected_table.where(
                    selection_col == gold_set.name
                ).count()
                query = selected_table.where(selection_col == None)  # noqa: E711
                if query.count() == 0 and already_in_set_count == 0:
                    raise ValueError(
                        f"Not enough data to split among {len(self._sets)} sets."
                    )
                query.update(  # noqa: E711
                    {self.selector.selection_key: gold_set.name}
                )

        split_table = selected_table
        if self.in_described_table:
            if not isinstance(description, Table):
                raise ValueError(
                    "in_described_table is set to True, but description is not a PixelTable Table."
                )
            for set_cfg in self._sets:
                set_name = set_cfg.name
                set_indices = self.selector.get_selected_sample_indices(
                    selected_table,
                    selection_key=self.selector.selection_key,
                    value=set_name,
                )
                if self.selector.selection_key not in description.columns():
                    description.add_column(
                        if_exists="error",
                        **{self.selector.selection_key: pxt.String},
                    )
                set_value_to_idx_rows(
                    description,
                    col_expr=get_expr_from_column_name(
                        description, self.selector.selection_key
                    ),
                    idx_expr=description.idx,
                    indices=set_indices,
                    value=set_name,
                )
            split_table = description

        self._drop_tables(drop_all=False)

        return split_table

    def _drop_tables(self, drop_all: bool = True) -> None:
        """Drop all intermediate tables created during the splitting process.

        Args:
            drop_all: Whether to drop all tables including the selected table.
            The selected table is either the selector's table or the descriptor's table
            depending on the `in_described_table` flag.
        """
        if self.drop_table:
            table_names = [
                self.selector.table_path,
                *([self.descriptor.table_path] if self.descriptor is not None else []),
                *([self.vectorizer.table_path] if self.vectorizer is not None else []),
            ]

            to_not_drop = (
                self.descriptor.table_path
                if self.in_described_table and self.descriptor is not None
                else self.selector.table_path
            )

            for table in table_names:
                if table != to_not_drop or drop_all:
                    pxt.drop_table(table, if_not_exists="ignore")
