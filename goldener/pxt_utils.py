from typing import Literal, Any

import pixeltable as pxt
from pixeltable.catalog import Table
from pixeltable.exprs import Expr


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


def create_views_per_column_value(
    table: Table,
    col_expr: Expr,
    if_exists: Literal["error", "ignore", "replace", "replace_force"] = "error",
) -> list[str]:
    """Create views for each distinct value in a specified column of a PixelTable table.

    Args:
        table: The PixelTable table.
        col_expr: The column expression to create views for.
        if_exists: Behavior when the view already exists. Options are "error", "ignore",
            "replace", and "replace_force". Default is "error".

    Returns:
        A list of view path corresponding to each label.
    """
    view_name_list = []
    table_path_split = table.get_metadata()["path"].split(".")
    table_dir = ".".join(table_path_split[:-1])
    table_name = table_path_split[-1]
    col_name = col_expr.display_str()
    for col_value_dict in col_expr.distinct().collect():
        col_value = col_value_dict[col_name]
        view_name = f"{table_dir}.{table_name}_{col_name}_{col_value}"
        pxt.create_view(
            view_name, table.where(col_expr == col_value), if_exists=if_exists
        )
        view_name_list.append(view_name)

    return view_name_list


def set_value_to_full_column(
    table: Table,
    col_expr: Expr,
    value: Any,
) -> None:
    """Set a column to a specific value for all rows in a PixelTable table.

    Args:
        table: The PixelTable table.
        col_expr: The column expression to be set.
        value: The value to set the column to.
    """
    table.update({col_expr.display_str(): value})


def set_value_to_idx_rows(
    table: Table,
    col_expr: Expr,
    idx_list: list[int],
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
        set_value_to_idx_rows(table, col_expr, indices, new_value)
