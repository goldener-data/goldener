from collections import defaultdict
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

import pixeltable as pxt

from goldener.torch_utils import collate_keeping_sequences_as_sequences


class GoldDoer:
    """Shared initialization for data-orchestration features.

    Note:
        This is the base class for features instantiating data orchestration.

    Attributes:
        doer_id: Stable identifier of the doer type, used in intermediate table metadata.
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
        """Initialize the GoldDoer.

        Args:
            allow_existing: Whether to allow using existing tables. Defaults to True.
            drop_table: Whether to drop intermediate tables. Defaults to False.
            max_batches: Optional maximum number of batches to process.
        """
        self._doer_id = str(
            uuid5(NAMESPACE_URL, f"{type(self).__module__}.{type(self).__qualname__}")
        )
        self.allow_existing = allow_existing
        self.drop_table = drop_table
        self.max_batches = max_batches

    @property
    def doer_id(self) -> str:
        """Return the stable identifier used in intermediate table metadata."""
        return self._doer_id


class GoldDoerWithTable(GoldDoer):
    """Shared initialization for data-orchestration features that populate intermediate tables.

    The column names of the table are validated at initialization.
    `_RESERVED_KEYS` holds `idx` and `idx_vector`. Those names are merged with
    the keys of `_MINIMAL_SCHEMA`, and that merged set is what collision
    checks treat as identity columns. A customizable key cannot use a name in
    the merged set. Customizable keys also cannot share one value across two
    attributes. `to_keep_schema` cannot redefine a name in the merged set.
    Child classes must set their own attributes before calling
    `super().__init__()`.

    Attributes:
        table_path: Path to the PixelTable table where results are stored.
        doer_id: Stable identifier of the doer type, used in intermediate table metadata.
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
        ValueError: If a customizable key uses a name in `_RESERVED_KEYS`
            merged with the keys of `_MINIMAL_SCHEMA`, if customizable keys
            share the same value, or if `to_keep_schema` defines one of those
            names.

    Note:
        This is the base class for features instantiating data orchestration that
        populate an intermediate table.  `_MINIMAL_SCHEMA` defines the minimal
        expected columns in the intermediary table. Each subclass should adapt it if needed.
        `idx_vector` stays reserved through `_RESERVED_KEYS` when a subclass schema
        does not list it.
    """

    _RESERVED_KEYS: frozenset[str] = frozenset({"idx", "idx_vector"})
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
        """Initialize the GoldDoerWithTable.

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
            ValueError: If a key attribute is named after an identity column, if key
                attributes share the same value, or if `to_keep_schema` defines an
                identity column.
        """
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
        """Return `_MINIMAL_SCHEMA` keys merged with `_RESERVED_KEYS`."""
        return set(self._MINIMAL_SCHEMA) | set(self._RESERVED_KEYS)

    def _validate_key_attributes(self) -> None:
        """Reject customizable keys that use a merged identity name.

        Raises:
            ValueError: If a customizable key equals a name in
                `_RESERVED_KEYS` merged with the keys of `_MINIMAL_SCHEMA`.
        """
        identity_columns = self._identity_column_names()
        for name, value in self._key_attributes().items():
            if value in identity_columns:
                raise ValueError(
                    f"{name}='{value}' collides with the identity column '{value}'."
                )

    def _validate_column_names(self) -> None:
        """Check merged identity names, duplicate keys, and kept columns.

        Customizable keys are checked by `_validate_key_attributes`. The
        identity set is the keys of `_MINIMAL_SCHEMA` merged with
        `_RESERVED_KEYS`, so `idx` and `idx_vector` raise the identity-column
        error even when the schema lists only `idx`. If that check passes,
        duplicate customizable keys are reported together. `to_keep_schema`
        may share a name with an input key attribute (for example `label_key`
        and a kept label column). It cannot redefine a name in the merged set.

        Raises:
            ValueError: If a customizable key uses a merged identity name, if
                customizable keys share the same value (all duplicates are
                reported at once), or if `to_keep_schema` defines a merged
                identity name.
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
