from dataclasses import dataclass

import pixeltable as pxt
from pixeltable.catalog import Table
from pixeltable.exprs import Expr
from torch.utils.data import Dataset

from goldener.describe import GoldDescriptor
from goldener.pxt_utils import (
    get_expr_from_column_name,
    pxt_torch_dataset_collate_fn,
    GoldPxtTorchDataset,
    get_array_column_shapes,
    _get_column_distinct_ratios,
    set_value_to_idx_rows,
)
from goldener.select import GoldSelector
from goldener.torch_utils import ResetableTorchIterableDataset
from goldener.utils import get_ratio_list_sum


@dataclass
class GoldSet:
    """Configuration for a gold set used for splitting.

    Attributes:
        name: Name of the gold set.
        ratio: Ratio of samples to assign to this set (between 0 and 1). This value
        cannot be one of 0 or 1 (goal is to select a subset of the full dataset).
    """

    name: str
    ratio: float

    def __post_init__(self) -> None:
        if not (0 < self.ratio < 1):
            raise ValueError("Ratio must be between 0 and 1.")


class GoldSplitter:
    """Splitter that divides a dataset into multiple sets based on features.

    When the sum of ratios of the provided sets during initialization is less than 1,
    an additional set named "not assigned" is added to cover the remaining samples.

    Attributes:
        sets: List of GoldSet configurations defining the splits.
        descriptor: GoldDescriptor used to describe the dataset.
        selector: GoldSelector used to select samples for each set. The collate_fn of the selector
        will be set to `pxt_torch_dataset_collate_fn`.
        class_key: Optional key for class-based stratification.
        drop_table: Whether to drop the described table after splitting.
    """

    def __init__(
        self,
        sets: list[GoldSet],
        descriptor: GoldDescriptor,
        selector: GoldSelector,
        class_key: str | None = None,
        drop_table: bool = False,
    ) -> None:
        """Initialize the GoldSplitter.

        Args:
            sets: List of GoldSet configurations defining the splits. When the sum of ratios is less than 1,
            an additional set named "not assigned" will be created to cover the remaining samples.
            descriptor: GoldDescriptor used to describe the dataset.
            selector: GoldSelector used to select samples for each set. The collate_fn of the selector
            will be set to `pxt_torch_dataset_collate_fn`.
            class_key: Optional key for class-based stratification.
            drop_table: Whether to drop the described table after splitting.

        Raises:
            ValueError: If set names are not unique or ratios do not sum to 1.

        """
        self.descriptor = descriptor
        self.selector = selector
        self.drop_table = drop_table
        self.class_key = class_key

        ratios_sum = get_ratio_list_sum([s.ratio for s in sets])
        set_names = [s.name for s in sets]
        if len(set_names) != len(set(set_names)):
            raise ValueError(f"Set names must be unique, got {set_names}")

        if ratios_sum != 1.0:
            sets.append(GoldSet(name="not assigned", ratio=1.0 - ratios_sum))

        self.sets = sets

        self.selector.collate_fn = pxt_torch_dataset_collate_fn

    def split(self, dataset: Dataset) -> dict[str, set[int]]:
        """Split the dataset into multiple sets based on the configured ratios.

        The dataset is first described using the gold descriptor (extracts features), and then samples are selected
        for each set based on the specified ratios. If a class_key is provided, stratified sampling is performed
        to maintain class distribution across sets.

        The last set and potentially the last class of each set in case of stratification will take the remaining samples
        to avoid rounding issues.

        Args:
            dataset: Dataset to be split.

        Returns:
            A dictionary mapping set names to sets of sample indices assigned to each set.

        Raises:
            ValueError: If any set results in zero samples due to its ratio, if class_key is not found,
            or if class stratification results in zero samples for any class in a set.
        """
        described_table = self.descriptor.describe(dataset)
        described_table.add_column(gold_set=pxt.String, if_exists="error")

        class_expr = self._get_class_expr(described_table)
        class_ratios = _get_column_distinct_ratios(described_table, class_expr)

        sample_count = described_table.count()

        # select data for all sets except the last one
        for idx_set, gold_set in enumerate(self.sets[:-1]):
            set_count = int(gold_set.ratio * sample_count)
            if set_count == 0:
                raise ValueError(
                    f"Set '{gold_set.name}' has ratio {gold_set.ratio} which results "
                    f"in zero samples for dataset of size {sample_count}."
                )
            self._select_for_set(
                described_table,
                class_ratios,
                gold_set.name,
                set_count,
                class_expr,
            )

        # remaining samples are assigned to the last set
        remaining_idx_list = [
            row["idx"]
            for row in described_table.select(described_table.idx)
            .where(described_table.gold_set == None)  # noqa: E711
            .collect()
        ]
        set_value_to_idx_rows(
            described_table,
            described_table.gold_set,
            set(remaining_idx_list),
            self.sets[-1].name,
        )

        self._drop_tables()

        return {
            gold_set.name: set(
                row["idx"]
                for row in described_table.where(
                    described_table.gold_set == gold_set.name
                )
                .select(described_table.idx)
                .collect()
            )
            for gold_set in self.sets
        }

    def _get_class_expr(self, described_table: pxt.Table) -> Expr:
        if (
            self.class_key is not None
            and self.class_key not in described_table.columns()
        ):
            raise ValueError(
                f"Class key '{self.class_key}' not found in described table columns: {described_table.columns()}"
            )

        class_key = self.class_key if self.class_key is not None else "gold_split_class"
        if self.class_key is None:
            described_table.add_column(**{class_key: pxt.String}, if_exists="error")

        return get_expr_from_column_name(described_table, class_key)

    def _select_for_set(
        self,
        described_table: Table,
        class_ratios: dict[str, float],
        set_name: str,
        set_count: int,
        class_expr: Expr,
    ) -> None:
        for class_idx, (class_label, class_ratio) in enumerate(class_ratios.items()):
            if class_idx < len(class_ratios) - 1:
                class_count = int(set_count * class_ratio)
                if class_count == 0:
                    raise ValueError(
                        f"Class '{class_label}' has ratio {class_ratio} which results "
                        f"in zero samples for set '{set_name}' with size {set_count}."
                    )
            else:
                # The last class takes the missing samples count to avoid rounding issues
                existing_count = described_table.where(
                    described_table.gold_set == set_name
                ).count()
                class_count = set_count - existing_count

            class_table = described_table.where(
                (class_expr == class_label) & (described_table.gold_set == None)  # noqa: E711
            ).select()

            torch_dataset = ResetableTorchIterableDataset(
                GoldPxtTorchDataset(
                    class_table, shapes=get_array_column_shapes(described_table)
                )
            )

            selected_indices = self.selector.select(torch_dataset, class_count)
            described_table.where(described_table.idx.isin(selected_indices)).update(
                {"gold_set": set_name}
            )

    def _drop_tables(self) -> None:
        if self.drop_table:
            for table_name in (self.descriptor.table_path, self.selector.table_path):
                pxt.drop_table(table_name)
