from time import sleep

import numpy as np
import pytest

import torch
from coreax.kernels import LinearKernel
from pixeltable import Error
from sklearn.decomposition import PCA

import pixeltable as pxt
from torch.utils.data import Dataset

from goldener.pxt_utils import GoldPxtTorchDataset, pxt_torch_dataset_collate_fn
from goldener.reduce import GoldSKLearnReductionTool
from goldener.select import (
    GoldSelector,
    GoldGreedyFarthestPointSelectionTool,
    GoldGreedyKCenterSelectionTool,
    DistanceType,
)
from goldener.select import (
    GoldGreedyClosestPointSelectionTool,
    GoldGreedyKernelPointsSelectionTool,
    GoldZCoreSelectionTool,
)


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
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def test_distribute_cannot_be_enabled(self):
        error = "Distributed processing is not implemented for GoldSelector"
        with pytest.raises(NotImplementedError, match=error):
            GoldSelector(
                table_path="unit_test.test_distribute",
                distribute=True,
            )

        selector = GoldSelector(table_path="unit_test.test_distribute")
        assert selector.distribute is False

        with pytest.raises(NotImplementedError, match=error):
            selector.distribute = True
        assert selector.distribute is False

    def test_collate_fn_defaults_to_pxt_torch_dataset_collate_fn(self):
        selector = GoldSelector(table_path="unit_test.select_default_collate")
        assert selector.collate_fn is pxt_torch_dataset_collate_fn

        def custom_collate_fn(batch):
            return batch

        selector = GoldSelector(
            table_path="unit_test.select_default_collate",
            collate_fn=custom_collate_fn,
        )
        assert selector.collate_fn is custom_collate_fn

    def test_selection_table_creation_from_table(self):
        src_path = "unit_test.src_table_input"
        desc_path = "unit_test.test_select_from_table"

        source_rows = [
            {
                "idx": 0,
                "vectorized": torch.zeros(1, 5).numpy(),
                "label": "dummy",
                "idx_vector": 0,
            },
            {
                "idx_vector": 1,
                "vectorized": torch.zeros(1, 5).numpy(),
                "label": "dummy",
                "idx": 0,
            },
        ]

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
            "idx_vector",
        }
        row_indices = [
            row["idx_vector"]
            for row in pxt_table.select(pxt_table.idx_vector).collect()
        ]
        assert set(row_indices) == {0, 1}

    def test_selection_table_from_table_when_missing_vectorized(self):
        src_path = "unit_test.src_table_invalid"
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

    def test_selection_table_from_table_when_invalid_old(self):
        src_path = "unit_test.src_table_invalid"
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

    def test_selection_table_from_dataset(self):
        table_path = "unit_test.test_select_initialize"

        sample = {
            "vectorized": torch.rand(5),
            "idx": 0,
        }
        dataset = DummyDataset([sample, sample])

        selector = GoldSelector(table_path=table_path, allow_existing=False)

        pxt_table = selector._selection_table_from_dataset(
            dataset, old_selection_table=None
        )

        assert set(pxt_table.columns()) == {
            selector.selection_key,
            selector.vectorized_key,
            "idx",
            "idx_vector",
        }
        row_indices = [
            row["idx_vector"]
            for row in pxt_table.select(pxt_table.idx_vector).collect()
        ]
        assert set(row_indices) == {0, 1}

    def test_select_in_table_from_dataset(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=None
        )

        selection_table = selector.select_in_table(
            dataset, select_size=10, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

        # validate running with already filled work
        selector.select_in_table(dataset, select_size=10, value="train")

    def test_select_in_table_with_new_idx_vector(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [
                {"vectorized": torch.rand(5), "idx": idx, "idx_vector": idx}
                for idx in range(100)
            ]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=None
        )

        selection_table = selector.select_in_table(
            dataset, select_size=10, value="train"
        )
        selected_indices_1 = selector.get_selection_indices(
            selection_table, "train", selector.selection_key
        )
        assert len(selected_indices_1) == 10

        dataset = DummyDataset(
            [
                {"vectorized": torch.rand(5), "idx": idx, "idx_vector": 100 + idx}
                for idx in range(100)
            ]
        )
        selection_table = selector.select_in_table(
            dataset, select_size=20, value="train"
        )

        selected_indices_2 = selector.get_selection_indices(
            selection_table, "train", selector.selection_key
        )
        assert len(selected_indices_2) == 20
        assert selected_indices_1.issubset(selected_indices_2)

    def test_select_in_table_with_wrong_size(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=None
        )

        with pytest.raises(
            ValueError, match="When select_size is a float, it must be in the range"
        ):
            selector.select_in_table(dataset, select_size=1.1, value="train")

        with pytest.raises(ValueError, match="select_size must be a positive integer"):
            selector.select_in_table(dataset, select_size=0, value="train")

    def test_select_in_table_from_dataset_with_ratio(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=None
        )

        selection_table = selector.select_in_table(
            dataset, select_size=0.1, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

    def test_select_in_table_from_dataset_with_small_ratio(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx_sample": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=None
        )

        selection_table = selector.select_in_table(
            dataset, select_size=0.0001, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 1
        )

    def test_select_in_table_from_dataset_with_class(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [
                {
                    "vectorized": torch.rand(5),
                    "idx": idx % 200,
                    "label": str(idx % 2),
                    "idx_vector": idx,
                }
                for idx in range(1000)
            ]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=True,
            label_key="label",
            batch_size=10,
            max_batches=None,
        )

        selection_table = selector.select_in_table(
            dataset,
            select_size=100,
            value="train",
        )

        assert selection_table.count() == 1000
        assert (
            len(
                selector.get_selection_indices(
                    selection_table,
                    "train",
                    selector.selection_key,
                    label_key=selector.label_key,
                    label_value="0",
                )
            )
            == 50
        )
        assert (
            len(
                selector.get_selection_indices(
                    selection_table,
                    "train",
                    selector.selection_key,
                    label_key=selector.label_key,
                    label_value="1",
                )
            )
            == 50
        )

    def test_select_in_table_from_dataset_with_excluded_labels(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [
                {
                    "vectorized": torch.rand(5),
                    "idx": idx,
                    "label": "excluded" if idx < 50 else "kept",
                    "idx_vector": idx,
                }
                for idx in range(100)
            ]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=True,
            label_key="label",
            exclude_labels={"excluded"},
            batch_size=10,
        )

        selection_table = selector.select_in_table(
            dataset,
            select_size=10,
            value="train",
        )

        assert selection_table.count() == 50
        for row in selection_table.collect():
            assert row["label"] == "kept"

    def test_select_with_exclude_labels_without_label_key_raises(self):
        with pytest.raises(
            ValueError,
            match="If exclude_labels is provided, label_key must also be provided",
        ):
            GoldSelector(
                table_path="unit_test.test_select",
                exclude_labels={"excluded"},
            )

    def test_select_in_table_with_chunk(self):
        table_path = "unit_test.test_select_chunk"

        dataset = DummyDataset(
            [
                {
                    "vectorized": torch.rand(
                        5,
                    ),
                    "idx": idx,
                }
                for idx in range(100)
            ]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, chunk=26, batch_size=10
        )

        selection_table = selector.select_in_table(
            dataset, select_size=10, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

    def test_select_in_table_with_reducer(self):
        table_path = "unit_test.test_select_reducer"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=True,
            reducer=GoldSKLearnReductionTool(PCA(n_components=3)),
            batch_size=10,
        )

        selection_table = selector.select_in_table(
            dataset, select_size=10, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

    def test_select_in_table_with_max_batches(self):
        table_path = "unit_test.test_select_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=2
        )

        selection_table = selector.select_in_table(
            dataset, select_size=10, value="train"
        )

        assert selection_table.count() == 20
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 10
        )

    def test_select_in_table_with_restart_and_reducer(self):
        table_path = "unit_test.test_select_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=True,
            batch_size=10,
            reducer=GoldSKLearnReductionTool(PCA(n_components=3)),
            max_batches=2,
        )

        selector.select_in_table(dataset, select_size=10, value="train")

        selector.max_batches = None
        selection_table = selector.select_in_table(
            dataset, select_size=20, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 20
        )

    def test_select_in_table_with_restart(self):
        table_path = "unit_test.test_select_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=True, batch_size=10, max_batches=2
        )

        selector.select_in_table(dataset, select_size=10, value="train")

        selector.max_batches = None
        selection_table = selector.select_in_table(
            dataset, select_size=20, value="train"
        )

        assert selection_table.count() == 100
        assert (
            selection_table.where(
                selection_table[selector.selection_key] == "train"
            ).count()
            == 20
        )

    def test_select_in_table_with_restart_disallowed(self):
        table_path = "unit_test.test_select_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=2
        )

        selector.select_in_table(dataset, select_size=10, value="train")

        selector.max_batches = None

        # calling select_in_table when a table exists and allow_existing is False should raise
        with pytest.raises(
            ValueError, match="already exists and allow_existing is set to"
        ):
            selector.select_in_table(dataset, select_size=20, value="train")

    def test_select_in_table_with_not_enough_sample(self):
        table_path = "unit_test.test_select_not_enough"
        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=2
        )

        # calling select_in_table when a table exists and allow_existing is False should raise
        with pytest.raises(
            ValueError, match="cannot be greater than the total number of samples"
        ):
            selector.select_in_table(dataset, select_size=21, value="train")

    def test_select_in_table_from_table(self):
        src_path = "unit_test.test_select_in_table"

        src_table = pxt.create_table(
            src_path,
            source=[
                {
                    "vectorized": torch.rand(5).numpy().astype(np.float32),
                    "idx": idx % 10,
                    "idx_vector": idx,
                }
                for idx in range(100)
            ],
            if_exists="replace_force",
            primary_key="idx_vector",
        )

        selector = GoldSelector(
            table_path="unit_test.test_select",
            allow_existing=True,
            batch_size=10,
            max_batches=2,
        )

        selector.select_in_table(src_table, select_size=3, value="train")

        selector.max_batches = None
        selection_table = selector.select_in_table(
            src_table, select_size=6, value="train"
        )

        assert selection_table.count() == 100
        assert (
            len(
                selector.get_selection_indices(
                    selection_table, "train", selector.selection_key
                )
            )
            == 6
        )

    def test_select_in_table_from_table_with_vectorized_included(self):
        src_path = "unit_test.test_select_in_table"

        src_table = pxt.create_table(
            src_path,
            source=[
                {
                    "vectorized": torch.rand(5).numpy().astype(np.float32),
                    "idx": idx % 10,
                    "idx_vector": idx,
                }
                for idx in range(100)
            ],
            if_exists="replace_force",
            primary_key="idx_vector",
        )

        selector = GoldSelector(
            table_path="unit_test.test_select",
            allow_existing=True,
            batch_size=10,
            max_batches=None,
            include_vectorized_in_table=True,
        )

        selection_table = selector.select_in_table(
            src_table, select_size=6, value="train"
        )

        assert selection_table.count() == 100
        assert (
            len(
                selector.get_selection_indices(
                    selection_table, "train", selector.selection_key
                )
            )
            == 6
        )

    def test_select_in_dataset_from_dataset(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=2
        )

        dataset = selector.select_in_dataset(dataset, select_size=3, value="train")

        assert isinstance(dataset, GoldPxtTorchDataset)

        selected_count = 0
        total_count = 0
        already_selected = set()
        for item in dataset:
            total_count += 1
            if item["selected"] == "train" and item["idx"] not in already_selected:
                selected_count += 1
                already_selected.add(item["idx"])

        assert total_count == 20
        assert selected_count == 3

        dataset.keep_cache = False

    def test_select_from_dataset_with_restrict_to(self):
        table_path = "unit_test.test_restrict_to"

        dataset = DummyDataset(
            [
                {"vectorized": torch.rand(5), "idx": idx, "idx_vector": idx}
                for idx in range(20)
            ]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=None
        )

        dataset = selector.select_in_dataset(
            dataset, select_size=3, value="train", restrict_to={0, 1, 2, 3, 4}
        )

        selected_indices = set()
        for item in dataset:
            if item["selected"] == "train":
                selected_indices.add(item["idx_vector"])

        assert len(selected_indices) == 3
        assert selected_indices.issubset({0, 1, 2, 3, 4})

        dataset.keep_cache = False

    def test_select_from_table_with_restrict_to(self):
        src_path = "unit_test.src_table_restrict"
        src_table = pxt.create_table(
            src_path,
            source=[
                {
                    "vectorized": torch.rand(5).numpy().astype(np.float32),
                    "idx": idx,
                    "idx_vector": idx,
                }
                for idx in range(10)
            ],
            if_exists="replace_force",
            primary_key="idx_vector",
        )

        selector = GoldSelector(
            table_path="unit_test.test_select_from_table_restrict", allow_existing=True
        )

        result_table = selector.select_in_table(
            src_table, select_size=2, value="train", restrict_to={2, 5, 7}
        )

        assert result_table.count() == 10  # full table preserved

        selected_indices = selector.get_selection_indices(
            result_table, "train", selector.selection_key
        )
        assert len(selected_indices) == 2
        assert selected_indices.issubset({2, 5, 7})

    def test_select_in_dataset_with_restriction_idx_key(self):
        table_path = "unit_test.test_restriction_idx_key"

        dataset = DummyDataset(
            [
                {"vectorized": torch.rand(5), "idx": idx, "idx_vector": idx + 100}
                for idx in range(20)
            ]
        )
        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=None
        )
        dataset = selector.select_in_dataset(
            dataset,
            select_size=3,
            value="train",
            restrict_to={0, 1, 4},
            restriction_idx_key="idx",
        )
        selected_indices = set()
        selected_idx_vector = set()
        for item in dataset:
            if item["selected"] == "train":
                assert item["idx_vector"] == item["idx"] + 100
                selected_indices.add(item["idx"])
                selected_idx_vector.add(item["idx_vector"])

        assert len(selected_indices) == 3
        assert selected_indices.issubset({0, 1, 4})
        assert selected_idx_vector.issubset({100, 101, 104})

        dataset.keep_cache = False

    def test_select_in_dataset_with_restrict_to_exceeding_size(self):
        table_path = "unit_test.test_restrict_to"

        dataset = DummyDataset(
            [
                {"vectorized": torch.rand(5), "idx": idx, "idx_vector": idx}
                for idx in range(20)
            ]
        )
        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=None
        )

        with pytest.raises(
            ValueError, match="cannot be greater than the total number of samples"
        ):
            selector.select_in_dataset(
                dataset, select_size=5, value="train", restrict_to={0, 1, 2}
            )

    def test_select_in_table_from_dataset_with_already_selected(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [
                {
                    "vectorized": torch.rand(5),
                    "idx": idx,
                    "selected": "train" if idx % 10 else "val",
                }
                for idx in range(100)
            ]
        )

        selector = GoldSelector(
            table_path=table_path, allow_existing=False, batch_size=10, max_batches=2
        )

        with pytest.raises(
            ValueError,
            match=" selected samples for train, which is more than the requested",
        ):
            selector.select_in_table(dataset, select_size=15, value="train")

    def test_select_in_dataset_with_drop_table(self):
        table_path = "unit_test.test_select_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(5), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            allow_existing=False,
            batch_size=10,
            max_batches=2,
            drop_table=True,
        )

        dataset = selector.select_in_dataset(dataset, select_size=3, value="train")

        sleep(1)
        with pytest.raises(
            Error, match="Path 'unit_test.test_select_from_dataset' does not exist"
        ):
            pxt.get_table(table_path, if_not_exists="error")

        dataset.keep_cache = False

    def test_get_selected_indices(self):
        src_path = "unit_test.test_select"

        src_table = pxt.create_table(
            src_path,
            source=[
                {
                    "vectorized": torch.rand(5).numpy().astype(np.float32),
                    "idx": idx,
                    "selected": "train" if idx < 50 else None,
                    "label": "value" if idx < 25 else "other",
                }
                for idx in range(100)
            ],
            if_exists="replace_force",
        )

        sample_indices = GoldSelector.get_selection_indices(
            table=src_table,
            value="train",
            selection_key="selected",
        )

        assert sample_indices == set(range(50))
        assert len(sample_indices) == GoldSelector.get_selection_count(
            table=src_table, value="train", selection_key="selected"
        )

        sample_indices = GoldSelector.get_selection_indices(
            table=src_table,
            value=None,
            selection_key="selected",
        )
        assert sample_indices == set(range(50, 100, 1))

        sample_indices = GoldSelector.get_selection_indices(
            table=src_table,
            value="train",
            selection_key="selected",
            label_key="label",
            label_value="value",
        )
        assert sample_indices == set(range(25))

        sample_indices = GoldSelector.get_selection_indices(
            table=src_table,
            value="train",
            selection_key="selected",
            label_key="label",
            label_value="other",
        )
        assert sample_indices == set(range(25, 50, 1))

        with pytest.raises(
            ValueError, match="label_key and label_value must be set together"
        ):
            GoldSelector.get_selection_indices(
                table=src_table,
                value="train",
                selection_key="selected",
                label_key="label",
                label_value=None,
            )

        with pytest.raises(
            ValueError, match="label_key and label_value must be set together"
        ):
            GoldSelector.get_selection_indices(
                table=src_table,
                value="train",
                selection_key="selected",
                label_key=None,
                label_value="other",
            )


class TestGoldGreedyClosestPointSelectionTool:
    def test_simple_selection(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyClosestPointSelectionTool(device="cpu")
        indices = tool.select(x, k=2)

        assert indices == [0, 1]

    def test_with_cosine_distance(self) -> None:
        x = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0], [1.0, 0.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyClosestPointSelectionTool(
            device="cpu", distance=DistanceType.COSINE
        )
        indices = tool.select(x, k=2)

        assert indices == [0, 1]

    def test_select_all_points(self) -> None:
        x = torch.tensor(
            [[0.0], [1.0], [2.0], [3.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyClosestPointSelectionTool(device="cpu")
        indices = tool.select(x, k=x.size(0))

        assert indices == [0, 1, 2, 3]

    def test_with_k_greater_than_size(self) -> None:
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        tool = GoldGreedyClosestPointSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="k cannot be greater than the number of data points in x"
        ):
            tool.select(x, k=5)

    def test_rejects_1d_tensor(self) -> None:
        tool = GoldGreedyClosestPointSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10), k=2)

    def test_rejects_3d_tensor(self) -> None:
        tool = GoldGreedyClosestPointSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10, 5, 3), k=2)


class TestGoldGreedyFarthestPointSelectionTool:
    def test_simple_selection(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.2, 2.2]],
            dtype=torch.float32,
        )

        tool = GoldGreedyFarthestPointSelectionTool(device="cpu")
        indices = tool.select(x, k=2)

        assert indices == [2, 0]

    def test_with_cosine(self) -> None:
        x = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0], [1.0, 0.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyFarthestPointSelectionTool(
            device="cpu", distance=DistanceType.COSINE
        )
        indices = tool.select(x, k=2)

        assert indices == [2, 0]

    def test_select_all_points(self) -> None:
        x = torch.tensor(
            [[0.0], [1.0], [2.0], [3.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyFarthestPointSelectionTool(device="cpu")
        indices = tool.select(x, k=x.size(0))

        assert indices == [0, 1, 2, 3]

    def test_with_k_greater_than_size(self) -> None:
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        tool = GoldGreedyFarthestPointSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="k cannot be greater than the number of data points in x"
        ):
            tool.select(x, k=5)

    def test_rejects_1d_tensor(self) -> None:
        tool = GoldGreedyFarthestPointSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10), k=2)

    def test_rejects_3d_tensor(self) -> None:
        tool = GoldGreedyFarthestPointSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10, 5, 3), k=2)


class TestGoldGreedyKCenterSelectionTool:
    def test_simple_selection(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyKCenterSelectionTool(device="cpu")
        indices = tool.select(x, k=2)

        assert indices == [3, 0]

    def test_cosine_distance(self) -> None:
        x = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0], [3.0, 0.0], [1.0, 0.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyKCenterSelectionTool(
            device="cpu", distance=DistanceType.COSINE
        )
        indices = tool.select(x, k=2)

        assert indices == [0, 2]

    def test_select_all_points(self) -> None:
        x = torch.tensor(
            [[0.0], [1.0], [2.0], [3.0]],
            dtype=torch.float32,
        )

        tool = GoldGreedyKCenterSelectionTool(device="cpu")
        indices = tool.select(x, k=x.size(0))

        assert indices == [0, 1, 2, 3]

    def test_with_k_greater_than_size(self) -> None:
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        tool = GoldGreedyKCenterSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="k cannot be greater than the number of data points in x"
        ):
            tool.select(x, k=5)

    def test_rejects_1d_tensor(self) -> None:
        tool = GoldGreedyKCenterSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10), k=2)

    def test_rejects_3d_tensor(self) -> None:
        tool = GoldGreedyKCenterSelectionTool(device="cpu")
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10, 5, 3), k=2)

    def test_selection_with_anchors(self) -> None:
        # 1D layout along x-axis: four points
        x = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [2.0, 0.0],
                [20.0, 0.0],
            ],
            dtype=torch.float32,
        )

        # Two anchors between the points of x
        anchors = torch.tensor(
            [
                [0.5, 0.0],
                [2.5, 0.0],
            ],
            dtype=torch.float32,
        )

        tool = GoldGreedyKCenterSelectionTool(device="cpu")
        k = 2
        indices = tool.select(x, k=k, anchors=anchors)

        # We must select k distinct indices
        assert len(indices) == k
        assert len(set(indices)) == k

        # Indices must refer to rows in x (0..x_len-1)
        assert all(0 <= idx < x.size(0) for idx in indices)

        # In this symmetric 1D case we expect the extremes to be selected
        assert set(indices) == {1, 3}


class TestGoldGreedyKernelPointsSelectionTool:
    def test_simple_usage(self) -> None:
        tool = GoldGreedyKernelPointsSelectionTool(
            feature_kernel=LinearKernel(output_scale=1, constant=0)
        )

        x = torch.arange(12, dtype=torch.float32).view(4, 3)
        k = 2

        indices = tool.select(x, k)

        assert len(indices) == k
        assert len(set(indices)) == k

    def test_select_all_points_with_linear_kernel(self) -> None:
        tool = GoldGreedyKernelPointsSelectionTool(
            feature_kernel=LinearKernel(output_scale=1, constant=0)
        )

        x = torch.arange(12, dtype=torch.float32).view(4, 3)
        k = 4

        indices = tool.select(x, k)

        assert len(indices) == k
        assert len(set(indices)) == k

    def test_with_k_greater_than_size(self) -> None:
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        tool = GoldGreedyKernelPointsSelectionTool(
            feature_kernel=LinearKernel(output_scale=1, constant=0)
        )
        with pytest.raises(ValueError, match="k cannot be greater"):
            tool.select(x, k=5)

    def test_rejects_1d_tensor(self) -> None:
        tool = GoldGreedyKernelPointsSelectionTool(
            feature_kernel=LinearKernel(output_scale=1, constant=0)
        )
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10), k=2)

    def test_rejects_3d_tensor(self) -> None:
        tool = GoldGreedyKernelPointsSelectionTool(
            feature_kernel=LinearKernel(output_scale=1, constant=0)
        )
        with pytest.raises(
            ValueError, match="GoldSelectionTool only accepts 2D tensors"
        ):
            tool.select(torch.randn(10, 5, 3), k=2)


class TestGoldZCoreSelectionTool:
    def test_select_basic(self):
        device = torch.device("cpu")
        tool = GoldZCoreSelectionTool(
            device=device,
            distance=DistanceType.EUCLIDEAN,
            n_dim_for_score=2,
            n_random_anchors=10,
            n_redundancy=3,
            redundancy_scale=2,
            random_state=0,
        )
        x = torch.stack(
            [
                torch.tensor([0.0, 0.0]),
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 1.0]),
                torch.tensor([1.0, 1.0]),
            ]
        )

        k = 2
        indices = tool.select(x, k=k)

        assert isinstance(indices, list)
        assert len(indices) == k
        assert len(set(indices)) == k
        assert all(0 <= idx < len(x) for idx in indices)

    def test_select_k_equals_len(self):
        device = torch.device("cpu")
        tool = GoldZCoreSelectionTool(device=device, n_random_anchors=5, n_redundancy=2)
        x = torch.randn(5, 3)

        indices = tool.select(x, k=len(x))

        assert sorted(indices) == list(range(len(x)))

    def test_select_k_greater_than_len_raises(self):
        device = torch.device("cpu")
        tool = GoldZCoreSelectionTool(device=device, n_random_anchors=5, n_redundancy=2)
        x = torch.randn(4, 3)

        with pytest.raises(
            ValueError, match="k cannot be greater than the number of data points in x"
        ):
            tool.select(x, k=len(x) + 1)

    def test_select_reproducible_with_random_state(self):
        device = torch.device("cpu")
        x = torch.randn(20, 4)

        tool1 = GoldZCoreSelectionTool(
            device=device,
            n_dim_for_score=2,
            n_random_anchors=15,
            n_redundancy=5,
            random_state=123,
        )
        tool2 = GoldZCoreSelectionTool(
            device=device,
            n_dim_for_score=2,
            n_random_anchors=15,
            n_redundancy=5,
            random_state=123,
        )

        k = 5
        indices1 = tool1.select(x, k=k)
        indices2 = tool2.select(x, k=k)

        assert indices1 == indices2

    def test_select_n_dim_for_score_exceeds_feature_dim(self):
        device = torch.device("cpu")
        x = torch.randn(10, 3)
        tool = GoldZCoreSelectionTool(
            device=device,
            n_dim_for_score=5,
            n_random_anchors=8,
            n_redundancy=3,
            random_state=0,
        )

        with pytest.raises(
            ValueError,
            match="n_dim_for_score cannot be greater",
        ):
            tool.select(x, k=3)

    def test_select_with_cosine_distance(self):
        device = torch.device("cpu")
        x = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0], [1.0, 0.0]],
            dtype=torch.float32,
        )

        tool = GoldZCoreSelectionTool(
            device=device,
            distance=DistanceType.COSINE,
            n_dim_for_score=2,
            n_random_anchors=2,
            n_redundancy=1,
            random_state=0,
        )
        indices = tool.select(x, k=2)

        assert len(indices) == 2

    def test_with_0_init(self):
        with pytest.raises(
            ValueError,
            match="n_dim_for_score must be a positive integer",
        ):
            GoldZCoreSelectionTool(
                device=torch.device("cpu"),
                n_dim_for_score=0,
                n_random_anchors=8,
                n_redundancy=3,
                random_state=0,
            )
        with pytest.raises(
            ValueError,
            match="n_random_anchors must be a positive integer",
        ):
            GoldZCoreSelectionTool(
                device=torch.device("cpu"),
                n_dim_for_score=2,
                n_random_anchors=0,
                n_redundancy=3,
                random_state=0,
            )

        with pytest.raises(
            ValueError,
            match="eps must be a positive float",
        ):
            GoldZCoreSelectionTool(
                device=torch.device("cpu"),
                n_dim_for_score=2,
                n_random_anchors=8,
                n_redundancy=3,
                random_state=0,
                eps=0,
            )
