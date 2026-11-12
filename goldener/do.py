from typing import Callable

import pixeltable as pxt

from goldener.torch_utils import collate_keeping_sequences_as_sequences


class GoldDoer:
    """Shared initialization for Goldener's data orchestration tools.

    Args:
        allow_existing: Whether to allow using existing tables. Defaults to True.
        drop_table: Whether to drop intermediate tables. Defaults to False.
        max_batches: Optional maximum number of batches to process.
    """

    def __init__(
        self,
        allow_existing: bool = True,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        self.allow_existing = allow_existing
        self.drop_table = drop_table
        self.max_batches = max_batches


class GoldDoerWithTable(GoldDoer):
    """Shared initialization for GoldDoers that populate intermediate tables.

    The column names of the table are validated at initialization: the identity
    columns of `_MINIMAL_SCHEMA`, the values of every attribute whose name contains
    `key` (e.g. `data_key`, `label_key`) and the columns of `to_keep_schema` must not
    collide. Child classes must therefore set their own attributes before calling
    `super().__init__()`, so that the validation sees them.

    Args:
        allow_existing: Whether to allow using existing tables. Defaults to True.
        drop_table: Whether to drop intermediate tables. Defaults to False.
        max_batches: Optional maximum number of batches to process.
        collate_fn: Function to collate dataset samples into batches. If None,
            `collate_keeping_sequences_as_sequences` is used.
        to_keep_schema: Optional schema for additional columns to preserve.
        min_pxt_insert_size: Minimum number of rows to accumulate before inserting
            into PixelTable. Defaults to 100.
        batch_size: Batch size used when iterating over the data. Defaults to 1.
        num_workers: Number of workers for the PyTorch DataLoader. Defaults to 0.

    Raises:
        ValueError: If a key attribute is named after an identity column, if two key
            attributes share the same value, or if `to_keep_schema` defines an
            identity column.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        allow_existing: bool = True,
        drop_table: bool = False,
        max_batches: int | None = None,
        collate_fn: Callable | None = None,
        to_keep_schema: dict[str, type] | None = None,
        min_pxt_insert_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            allow_existing=allow_existing,
            drop_table=drop_table,
            max_batches=max_batches,
        )
        self.collate_fn = (
            collate_fn
            if collate_fn is not None
            else collate_keeping_sequences_as_sequences
        )
        self.to_keep_schema = to_keep_schema
        self.min_pxt_insert_size = min_pxt_insert_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._validate_column_names()

    def _get_minimal_schema(self) -> dict[str, type]:
        """Return a copy of the identity columns of the table and their types.

        Returns:
            A new dictionary mapping identity column names to PixelTable types.
        """
        return self._MINIMAL_SCHEMA.copy()

    def _validate_column_names(self) -> None:
        """Check that identity columns, key attributes and kept columns do not collide.

        Key attributes are the instance attributes whose name contains `key` and whose
        value is a string; None values are ignored. Kept columns may share a name with
        an input key attribute (e.g. `label_key` and a kept label column).

        Raises:
            ValueError: If a key attribute is named after an identity column, if two key
                attributes share the same value, or if `to_keep_schema` defines an
                identity column.
        """
        identity_columns = set(self._get_minimal_schema())
        key_attributes = {
            name: value
            for name, value in vars(self).items()
            if "key" in name and isinstance(value, str)
        }

        for name, value in key_attributes.items():
            if value in identity_columns:
                raise ValueError(
                    f"{name}='{value}' collides with the identity column '{value}'."
                )

        attributes_by_value: dict[str, list[str]] = {}
        for name, value in key_attributes.items():
            attributes_by_value.setdefault(value, []).append(name)
        for value, names in attributes_by_value.items():
            if len(names) > 1:
                raise ValueError(
                    f"{', '.join(names)} share the same column name '{value}'."
                )

        if self.to_keep_schema is not None:
            collisions = sorted(identity_columns & set(self.to_keep_schema))
            if collisions:
                raise ValueError(
                    f"to_keep_schema cannot redefine the identity columns {collisions}."
                )
