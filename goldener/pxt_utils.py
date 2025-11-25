from collections import defaultdict
from typing import Literal, Any, Iterator, Sequence, Callable
import shutil

import numpy as np
import pixeltable as pxt
from pixeltable import Query
from pixeltable.catalog import Table
from pixeltable.exprs import Expr

import torch
from pixeltable.type_system import ArrayType
from pixeltable.utils.pytorch import PixeltablePytorchDataset

from goldener.utils import get_ratios_for_counts


def get_expr_from_column_name(table: Table, column_name: str) -> Expr:
    """Get the expression object for a given column name in a PixelTable table.

    Args:
        table: The PixelTable table.
        column_name: The name of the column.

    Returns:
        pxt.Expr: The expression object corresponding to the column name.

    Raises:
        ValueError: If the column name does not exist in the table schema.
    """
    if column_name not in table.columns():
        raise ValueError(f"Column '{column_name}' does not exist in the table schema.")

    return getattr(table, column_name)


def create_pxt_dirs_for_path(table_path: str) -> None:
    """Create necessary PixelTable directories for a given table path.

    Args:
        table_path: The full path of the PixelTable table (e.g., 'dir1.dir2.table_name').
    """
    pxt_path_split = table_path.split(".")
    for pxt_dir_idx in range(len(pxt_path_split) - 1):
        pxt_dir = ".".join(pxt_path_split[0 : pxt_dir_idx + 1])
        pxt.create_dir(pxt_dir, if_exists="ignore")


def create_pxt_table_from_sample(
    table_path: str,
    sample: dict,
    unwrap: bool = False,
    add: dict[str, Any] | None = None,
    if_exists: Literal["error", "replace_force"] = "error",
) -> Table:
    """Create a PixelTable table from a sample dictionary.

    Args:
        table_path: The full path of the PixelTable table to create (e.g., 'dir1.dir2.table_name').
        sample: A dictionary representing a single sample, where keys are column names and values are the corresponding data.
        unwrap: Whether to unwrap (only take first element) tensors/lists in the sample.
        add: Additional key-value pairs to add to the sample before creating the table.
        if_exists: Behavior if the table already exists. Options are 'error' or 'replace_force'.
    """
    create_pxt_dirs_for_path(table_path)

    if unwrap:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                # Only initial tensors will have a batch dimension added by the collate_fn
                # otherwise, they were initially single values (float or int)
                if value.ndim > 1:
                    sample[key] = value.squeeze(0)
                else:
                    sample[key] = value.item()
            else:
                # non tensor values are expected to be lists of single values
                sample[key] = value[0]

    if add is not None:
        for key, value in add.items():
            if key in sample:
                raise ValueError(f"Key '{key}' already exists in the sample.")
            sample[key] = value

    pxt_table = pxt.create_table(
        table_path,
        source=[
            {
                key: (value.numpy() if isinstance(value, torch.Tensor) else value)
                for key, value in sample.items()
            }
        ],
        if_exists=if_exists,
    )

    pxt_table.where(pxt_table.idx == 0).delete()  # remove the initial sample

    return pxt_table


def set_value_to_idx_rows(
    table: Table,
    col_expr: Expr,
    idx_list: set[int],
    value: int | float | str,
) -> None:
    """Set a column to a specific value for rows with given indices in a PixelTable table.

    Args:
        table: The PixelTable table. Must contain an 'idx' column.
        col_expr: The column expression to be set.
        idx_list: List of row indices to update.
        value: The value to set the column to.
    """
    table.where(table.idx.isin(idx_list)).update({col_expr.display_str(): value})


def update_column_if_too_many(
    table: Table,
    col_expr: Expr,
    value: int | float | str,
    max_count: int,
    new_value: Any,
) -> None:
    """Update some values of a column if the count of that value exceeds a maximum count.

    Args:
        table: The PixelTable table. Must contain an 'idx' column.
        col_expr: The column expression to be checked and updated.
        value: The value to check the count of.
        max_count: The maximum allowed count for the specified value.
        new_value: The new value to set for excess rows.
    """
    value_df = table.where(col_expr == value)
    value_count = value_df.select().count()
    if value_count > max_count:
        to_move = value_count - max_count
        indices = [row["idx"] for row in value_df.sample(to_move).collect()]
        set_value_to_idx_rows(table, col_expr, set(indices), new_value)


def pxt_torch_dataset_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for torch datasets obtained from a Pixeltable table.

    Args:
        batch: A list of samples, where each sample is a dictionary.

    Returns:
        A single dictionary with collated values. The numpy arrays and integers
        are stacked into torch tensors. All other types are kept as lists.
    """
    value_list_dict = defaultdict(list)
    conversion_dict: dict[str, Callable[[Sequence[Any]], torch.Tensor]] = {}

    def stack_arrays_and_convert_to_torch(arrays: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(arrays, axis=0))

    def stack_int_as_torch(ints: Sequence[int]) -> torch.Tensor:
        return torch.tensor(ints, dtype=torch.int64)

    for idx_sample, sample in enumerate(batch):
        for key, value in sample.items():
            value_list_dict[key].append(value)
            if idx_sample == 0:
                if isinstance(value, np.ndarray):
                    conversion_dict[key] = stack_arrays_and_convert_to_torch
                elif isinstance(value, int):
                    conversion_dict[key] = stack_int_as_torch

    return {
        key: conversion_dict[key](value) if key in conversion_dict else value
        for key, value in value_list_dict.items()
    }


class GoldPxtTorchDataset(PixeltablePytorchDataset):
    """A Pixeltable PyTorch Dataset that reshapes array columns to their original shapes.

    This class intends to solve a current issue in Pixeltable dataset where array columns
    are flattened when converted to PyTorch datasets. This class reshapes those columns
    back to their original shapes during iteration.

    Attributes:
        shapes: A dictionary mapping column names of arrays to their original shapes.
        keep_cache: Whether to keep the cached parquet files after the dataset is garbage collected.
        path: Path to directory containing cached parquet files.
        image_format: Image format ('np' or 'pt').
    """

    def __init__(self, source: Table | Query, keep_cache: bool = False) -> None:
        """Initialize the GoldPxtTorchDataset.

        Args:
            source: The PixelTable Table or Query to convert to a PyTorch dataset.
            keep_cache: Whether to keep the cached parquet files after the dataset is garbage collected.
        """

        self.shapes = (
            get_array_column_shapes_from_table(source)
            if isinstance(source, Table)
            else get_array_column_shapes_from_query(source)
        )
        self.keep_cache = keep_cache

        torch_dataset = source.to_pytorch_dataset()
        assert isinstance(torch_dataset, PixeltablePytorchDataset)
        super().__init__(torch_dataset.path, torch_dataset.image_format)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        super_iterator = super().__iter__()
        for item in super_iterator:
            yield {
                key: value
                if key not in self.shapes
                else value.reshape(self.shapes[key])
                for key, value in item.items()
            }

    def __del__(self) -> None:
        """Clean up cached parquet files when the dataset is destroyed.

        This method removes the cache directory created by PixeltablePytorchDataset
        to prevent accumulation of temporary files.
        """
        if self.path.exists() and not self.keep_cache:
            shutil.rmtree(self.path, ignore_errors=True)


def get_array_column_shapes_from_table(table: Table) -> dict[str, tuple[int, ...]]:
    """Get the shapes of array columns in a PixelTable table.

    Args:
        table: The PixelTable table.

    Returns:
        A dictionary mapping column names of array columns to their shapes.
    """
    shapes = {}
    for col_name in table.columns():
        col_expr = get_expr_from_column_name(table, col_name)
        col_expr_dict = col_expr.col_type.as_dict()
        if col_expr_dict["_classname"] == "ArrayType":
            shapes[col_name] = tuple(col_expr_dict["shape"])

    return shapes


def get_array_column_shapes_from_query(query: Query) -> dict[str, tuple[int, ...]]:
    """Get the shapes of array columns in a PixelTable Query.

    Args:
        query: The PixelTable Query.

    Returns:
        A dictionary mapping column names of array columns to their shapes.
    """
    shapes = {}

    for col_name, col_type in query.schema.items():
        if isinstance(col_type, ArrayType):
            shape = col_type.shape
            assert shape is not None
            shapes[col_name] = tuple(s for s in shape if s is not None)

    return shapes


def get_distinct_value_and_count_in_column(
    table: Table,
    col_expr: Expr,
) -> dict[Any, int]:
    """Get distinct values and their counts in a specified column of a PixelTable table.

    Args:
        table: The PixelTable table.
        col_expr: The column expression to analyze.

    Returns:
        A dictionary mapping distinct values to their counts in the specified column.
    """
    return {
        distinct_item[col_expr.display_str()]: table.where(
            col_expr == distinct_item[col_expr.display_str()]
        ).count()
        for distinct_item in table.select(col_expr).distinct().collect()
    }


def get_column_distinct_ratios(table: Table, class_expr: Expr) -> dict[str, float]:
    """Get the ratios of distinct values in a specified column of a PixelTable table.

    Args:
        table: The PixelTable table.
        class_expr: The column expression to analyze.

    Returns:
        A dictionary mapping distinct values to their ratios in the specified column.
    """
    value_and_count = get_distinct_value_and_count_in_column(table, class_expr)
    ratios = get_ratios_for_counts(value_and_count.values())

    return {
        class_label: class_ratio
        for class_label, class_ratio in zip(value_and_count.keys(), ratios)
    }


def is_view_of(view: Table, view_of: Table) -> bool:
    """Check if a PixelTable view is based on a specific table.

    Args:
        view: The PixelTable view.
        view_of: The PixelTable table the view is expected to be assigned to.

    Returns:
        True if the view is based on the specified table, False otherwise.
    """
    view_metadata = view.get_metadata()
    view_of_metadata = view_of.get_metadata()
    return (
        view_metadata["base"] == view_of_metadata["path"] and view_metadata["is_view"]
    )


def get_sample_row_from_idx(
    table: Table,
    idx: int = 0,
    collate_fn: Callable | None = None,
    expected_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Get a sample row from a PixelTable table.

    Args:
        table: The PixelTable table.
        idx: The index of the row to retrieve.
        collate_fn: An optional collate function to apply to the sample.
        expected_keys: An optional list of expected keys in the sample.

    Returns:
        A dictionary representing a sample row from the table.
    """
    sample_list = list(table.where(table.idx == idx).collect())

    if not sample_list:
        raise ValueError(f"No sample found at the index {idx}.")

    if len(sample_list) > 1:
        raise ValueError("Multiple samples found at the specified index.")

    sample = sample_list[0]
    if expected_keys is not None:
        not_present_keys = [key for key in expected_keys if key not in sample]
        if len(not_present_keys) > 0:
            raise ValueError(f"Sample is missing expected keys: {not_present_keys}")

    if collate_fn:
        sample = collate_fn([sample])

    return sample
