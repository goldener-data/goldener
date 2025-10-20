from typing import Literal

import pixeltable as pxt

from pixeltable.catalog import Table
from pixeltable.exprs import Expr

import torch


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
    if_exists: Literal["error", "replace_force"] = "error",
) -> Table:
    """Create a PixelTable table from a sample dictionary.

    Args:
        table_path: The full path of the PixelTable table to create (e.g., 'dir1.dir2.table_name').
        sample: A dictionary representing a single sample, where keys are column names and values are the corresponding data.
        if_exists: Behavior if the table already exists. Options are 'error' or 'replace_force'.
    """
    create_pxt_dirs_for_path(table_path)

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

    return pxt_table
