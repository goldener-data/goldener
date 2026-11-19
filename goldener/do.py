from collections import defaultdict
from typing import Callable

import pixeltable as pxt

from goldener.torch_utils import collate_keeping_sequences_as_sequences


class GoldDoer:
    """Shared initialization for Goldener's data orchestration tools.

    `RESERVED_KEYS` (`idx` and `idx_vector`) are reserved on every GoldDoer.
    Initialization rejects a customizable key that uses one of those names.
    A customizable key is a string attribute whose name contains `key`;
    `None` values are ignored. Subclasses set those attributes before calling
    `super().__init__()` so this check sees them.

    A reserved name that is also an identity column of this instance (a key of
    `_MINIMAL_SCHEMA` on a table doer) raises the identity-column error. A
    reserved name that is not in that schema, including `idx_vector` on a doer
    whose schema lists only `idx`, raises the reserved-key error. A doer
    without `_MINIMAL_SCHEMA` has no identity columns, so both names use the
    reserved-key error.

    Args:
        allow_existing: Whether to allow using existing tables. Defaults to True.
        drop_table: Whether to drop intermediate tables. Defaults to False.
        max_batches: Optional maximum number of batches to process.

    Raises:
        ValueError: If a customizable key uses a name in `RESERVED_KEYS`, or
            equals an identity column of this instance.
    """

    RESERVED_KEYS: frozenset[str] = frozenset({"idx", "idx_vector"})

    def __init__(
        self,
        allow_existing: bool = True,
        drop_table: bool = False,
        max_batches: int | None = None,
    ) -> None:
        self.allow_existing = allow_existing
        self.drop_table = drop_table
        self.max_batches = max_batches
        self._validate_key_attributes()

    def _key_attributes(self) -> dict[str, str]:
        """Return string attributes whose names contain `key`.

        `None` values are omitted. Non-string values are omitted.
        """
        return {
            name: value
            for name, value in vars(self).items()
            if "key" in name and isinstance(value, str)
        }

    def _identity_column_names(self) -> set[str]:
        """Return this instance's identity columns, or an empty set.

        Identity columns are the keys of `_MINIMAL_SCHEMA` when that class
        attribute exists. `GoldDoer` itself has no table schema.
        """
        schema: dict[str, type] | None = getattr(self, "_MINIMAL_SCHEMA", None)
        if schema is None:
            return set()
        return set(schema)

    def _validate_key_attributes(self) -> None:
        """Reject customizable keys that use a reserved or identity name.

        Raises:
            ValueError: If a customizable key equals an identity column, or
                equals a name in `GoldDoer.RESERVED_KEYS` that is not an
                identity column of this instance. When a name is both, only
                the identity-column error is raised.
        """
        identity_columns = self._identity_column_names()
        for name, value in self._key_attributes().items():
            if value in identity_columns:
                raise ValueError(
                    f"{name}='{value}' collides with the identity column '{value}'."
                )
            if value in GoldDoer.RESERVED_KEYS:
                raise ValueError(
                    f"{name}='{value}' is reserved for internal indexing "
                    "and cannot be used as a custom key."
                )


class GoldDoerWithTable(GoldDoer):
    """Shared initialization for GoldDoers that populate intermediate tables.

    The column names of the table are validated at initialization. Customizable
    keys cannot use `idx` or `idx_vector` (`GoldDoer.RESERVED_KEYS`), even when
    `_MINIMAL_SCHEMA` lists only `idx`. They also cannot repeat an identity
    column or share one value across two attributes. `to_keep_schema` cannot
    redefine an identity column or introduce a reserved name that is not in the
    schema. `label_key` is assigned in this constructor, after `GoldDoer`
    initialization, and is included in the same check. Child classes must set
    their own attributes before calling `super().__init__()`.

    Args:
        table_path: Path to the PixelTable table where results are stored.
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
        label_key: Optional key for label stratification.

    Raises:
        ValueError: If a customizable key uses a reserved name or an identity
            column, if customizable keys share the same value, or if
            `to_keep_schema` defines an identity column or a reserved name that
            is not in `_MINIMAL_SCHEMA`.

    Note:
        `_MINIMAL_SCHEMA` defines the minimal expected columns in the intermediary
        table. Each subclass should adapt it if needed. It does not by itself
        reserve `idx_vector`: that name is in `GoldDoer.RESERVED_KEYS`.
    """

    _MINIMAL_SCHEMA: dict[str, type] = {
        "idx": pxt.Required[pxt.Int],
    }

    def __init__(
        self,
        table_path: str,
        allow_existing: bool = True,
        drop_table: bool = False,
        max_batches: int | None = None,
        collate_fn: Callable | None = None,
        to_keep_schema: dict[str, type] | None = None,
        min_pxt_insert_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 0,
        label_key: str | None = None,
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
        self.table_path = table_path
        self.label_key = label_key
        self._validate_column_names()

    def _validate_column_names(self) -> None:
        """Check reserved keys, identity columns, duplicate keys, and kept columns.

        Customizable keys are checked by `_validate_key_attributes`. That reports
        an identity-column collision when the name is in `_MINIMAL_SCHEMA`, and
        a reserved-key error when the name is in `GoldDoer.RESERVED_KEYS` but
        not in this instance's schema. If that check passes, duplicate
        customizable keys are reported together. `to_keep_schema` may share a name with an input key
        attribute (for example `label_key` and a kept label column). It cannot
        redefine an identity column. A reserved name that is not an identity
        column, such as `idx_vector` when the schema lists only `idx`, is
        rejected separately.

        Raises:
            ValueError: If a customizable key uses a reserved or identity name,
                if customizable keys share the same value (all duplicates are
                reported at once), or if `to_keep_schema` defines an identity
                column or a reserved name absent from the schema.
        """
        self._validate_key_attributes()
        key_attributes = self._key_attributes()
        identity_columns = self._identity_column_names()

        attributes_by_value: dict[str, list[str]] = defaultdict(list)
        for name, value in key_attributes.items():
            attributes_by_value[value].append(name)
        duplicates = [
            f"'{value}' ({', '.join(names)})"
            for value, names in attributes_by_value.items()
            if len(names) > 1
        ]
        if duplicates:
            raise ValueError(
                f"Key attributes share the same column name: {'; '.join(duplicates)}."
            )

        if self.to_keep_schema is not None:
            kept_columns = set(self.to_keep_schema)
            collisions = sorted(identity_columns & kept_columns)
            if collisions:
                raise ValueError(
                    f"to_keep_schema cannot redefine the identity columns {collisions}."
                )
            reserved = sorted(
                (GoldDoer.RESERVED_KEYS & kept_columns) - identity_columns
            )
            if reserved:
                raise ValueError(
                    f"to_keep_schema cannot use the reserved columns {reserved}."
                )
