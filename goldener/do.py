from typing import Callable

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
    """

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
