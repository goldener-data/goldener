from time import sleep

import numpy as np
import pytest

import torch
from pixeltable import Error
from sklearn.decomposition import PCA

import pixeltable as pxt
from torch.utils.data import Dataset

from goldener.pxt_utils import GoldPxtTorchDataset
from goldener.reduce import GoldReducer
from goldener.select import GoldSelector


class DummyDataset(Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return (
            self._samples[idx].copy()
            if isinstance(self._samples[idx], dict)
            else self._samples[idx]
        )


class TestGoldSelector:
    def test_selection_table_creation_from_table(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_input"
        desc_path = "unit_test.test_select_from_table"

        source_rows = [
            {
                "idx": 0,
                "vectorized": torch.zeros(1, 5).numpy(),
                "label": "dummy",
                "idx_sample": 0,
            },
            {
                "idx": 1,
                "vectorized": torch.zeros(1, 5).numpy(),
                "label": "dummy",
                "idx_sample": 0,
            },
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        selector = GoldSelector(table_path=desc_path, allow_existing=False)

        pxt_table = selector._selection_table_from_table(
            src_table, old_selection_table=None
        )

        assert set(pxt_table.columns()) == {
            selector.selection_key,
            "idx",
            "idx_sample",
            "chunked",
        }
        row_indices = [row["idx"] for row in pxt_table.select(pxt_table.idx).collect()]
        assert set(row_indices) == {0, 1}

        pxt.drop_dir("unit_test", force=True)

    def test_selection_table_from_table_when_missing_vectorized(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_invalid"
        pxt.create_dir("unit_test", if_exists="ignore")

        src_table = pxt.create_table(
            src_path,
            source=[{"idx": 0, "notvec": [1, 2, 3]}],
            if_exists="replace_force",
        )

        selector = GoldSelector(table_path="unit_test.test_select", allow_existing=True)

        with pytest.raises(ValueError, match="does not contain the required column"):
            selector._selection_table_from_table(
                select_from=src_table, old_selection_table=None
            )

        pxt.drop_dir("unit_test", force=True)

    def test_selection_table_from_table_when_invalid_old(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_invalid"
        pxt.create_dir("unit_test", if_exists="ignore")

        src_table = pxt.create_table(
            src_path,
            source=[
                {
                    "idx": 0,
                }
            ],
            if_exists="replace_force",
        )

        selector = GoldSelector(table_path=src_path, allow_existing=True)

        with pytest.raises(ValueError, match="The table is missing required"):
            selector._selection_table_from_table(
                select_from=src_table, old_selection_table=src_table
            )

        pxt.drop_dir("unit_test", force=True)

    def test_selection_table_from_dataset(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_initialize"

        sample = {
            "vectorized": torch.rand(1, 5),
            "idx_sample": 7,
        }
        dataset = DummyDataset([sample, sample])

        selector = GoldSelector(table_path=table_path, allow_existing=False)

        pxt_table = selector._selection_table_from_dataset(
            dataset, old_selection_table=None
        )

        assert set(pxt_table.columns()) == {
            selector.vectorized_key,
            selector.selection_key,
            "idx",
            "idx_sample",
            "chunked",
        }
        row_indices = [row["idx"] for row in pxt_table.select(pxt_table.idx).collect()]
        assert set(row_indices) == {0, 1}

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_from_dataset(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=None
        )

        selection_table = selector.select_in_table(
            dataset, select_count=10, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_from_dataset_with_class(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [
                {"vectorized": torch.rand(5), "idx_sample": idx, "label": str(idx % 2)}
                for idx in range(100)
            ]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=True,
            class_key="label",
            batch_size=10,
            max_batches=None,
        )

        selection_table = selector.select_in_table(
            dataset,
            select_count=10,
            value="train",
        )

        assert selection_table.count() == 100
        assert (
            len(
                selector.get_selected_sample_indices(
                    selection_table,
                    "train",
                    selector.selection_key,
                    class_key=selector.class_key,
                    class_value="0",
                )
            )
            == 5
        )
        assert (
            len(
                selector.get_selected_sample_indices(
                    selection_table,
                    "train",
                    selector.selection_key,
                    class_key=selector.class_key,
                    class_value="1",
                )
            )
            == 5
        )

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_with_chunk(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_chunk"

        dataset = DummyDataset(
            [
                {
                    "vectorized": torch.rand(
                        5,
                    ),
                    "idx_sample": idx,
                }
                for idx in range(100)
            ]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, chunk=26, batch_size=10
        )

        selection_table = selector.select_in_table(
            dataset, select_count=10, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_with_reducer(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_reducer"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=True,
            reducer=GoldReducer(PCA(n_components=3)),
            batch_size=10,
        )

        selection_table = selector.select_in_table(
            dataset, select_count=10, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_with_max_batches(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=2
        )

        selection_table = selector.select_in_table(
            dataset, select_count=10, value="train"
        )

        assert selection_table.count() == 20
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_with_restart(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=2
        )

        selector.select_in_table(dataset, select_count=10, value="train")

        selector.max_batches = None
        selection_table = selector.select_in_table(
            dataset, select_count=20, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 20
        )

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_with_restart_disallowed(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=2
        )

        selector.select_in_table(dataset, select_count=10, value="train")

        selector.max_batches = None

        # calling select_in_table when a table exists and allow_existing is False should raise
        with pytest.raises(
            ValueError, match="already exists and allow_existing is set to"
        ):
            selector.select_in_table(dataset, select_count=20, value="train")

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_with_not_enough_sample(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_not_enough"
        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=2
        )

        # calling select_in_table when a table exists and allow_existing is False should raise
        with pytest.raises(
            ValueError, match="Cannot select more unique data points than available"
        ):
            selector.select_in_table(dataset, select_count=21, value="train")

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_table_from_table(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.test_select_in_table"

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path,
            source=[
                {
                    "vectorized": torch.rand(5).numpy().astype(np.float32),
                    "idx_sample": idx % 10,
                    "idx": idx,
                }
                for idx in range(100)
            ],
            if_exists="replace_force",
        )

        selector = GoldSelector(
            table_path="unit_test.test_select",
            allow_existing=True,
            batch_size=10,
            max_batches=2,
        )

        selector.select_in_table(src_table, select_count=3, value="train")

        selector.max_batches = None
        selection_table = selector.select_in_table(
            src_table, select_count=6, value="train"
        )

        assert selection_table.count() == 100
        assert (
            len(
                selector.get_selected_sample_indices(
                    selection_table, "train", selector.selection_key
                )
            )
            == 6
        )

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_dataset_from_dataset(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=2
        )

        dataset = selector.select_in_dataset(dataset, select_count=3, value="train")

        assert isinstance(dataset, GoldPxtTorchDataset)

        selected_count = 0
        total_count = 0
        already_selected = set()
        for item in dataset:
            total_count += 1
            if (
                item["selected"] == "train"
                and item["idx_sample"] not in already_selected
            ):
                selected_count += 1
                already_selected.add(item["idx_sample"])

        assert total_count == 20
        assert selected_count == 3

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)

    def test_select_in_dataset_with_drop_table(self):
        pxt.drop_dir("unit_test", force=True)

        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=False,
            batch_size=10,
            max_batches=2,
            drop_table=True,
        )

        dataset = selector.select_in_dataset(dataset, select_count=3, value="train")

        sleep(1)
        with pytest.raises(
            Error, match="Path 'unit_test.test_select_from_dataset' does not exist"
        ):
            pxt.get_table(table_path, if_not_exists="error")

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)
