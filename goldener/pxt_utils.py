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
