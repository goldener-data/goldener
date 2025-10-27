import pytest

import torch
from sklearn.decomposition import PCA

import pixeltable as pxt
from torch.utils.data import Dataset

from goldener.reduce import GoldReducer
from goldener.select import GoldSelector
from goldener.vectorize import GoldVectorizer


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


def collate_fn(batch):
    # simple collate that returns the single sample
    return batch[0]


class TestGoldSelector:
    def test_check_dataset_valid_no_collate(self):
        dataset = DummyDataset(
            [
                {"features": torch.tensor([1.0]), "idx": 0},
            ]
        )

        selector = GoldSelector(table_path="tmp.table", vectorizer=GoldVectorizer())
        # Should not raise
        selector._check_dataset(dataset)

    def test_check_dataset_valid_with_collate(self):
        dataset = DummyDataset(
            [
                {"features": torch.tensor([1.0]), "idx": 0},
            ]
        )

        selector = GoldSelector(
            table_path="tmp.table", vectorizer=GoldVectorizer(), collate_fn=collate_fn
        )
        # Should not raise
        selector._check_dataset(dataset)

    def test_check_dataset_non_dict(self):
        dataset = DummyDataset([1])
        selector = GoldSelector(table_path="tmp.table", vectorizer=GoldVectorizer())

        with pytest.raises(ValueError):
            selector._check_dataset(dataset)

    def test_check_dataset_missing_select_key(self):
        dataset = DummyDataset([{"not_features": torch.tensor([1.0]), "idx": 0}])
        selector = GoldSelector(table_path="tmp.table", vectorizer=GoldVectorizer())

        with pytest.raises(ValueError):
            selector._check_dataset(dataset)

    def test_check_dataset_select_not_tensor(self):
        dataset = DummyDataset([{"features": [1, 2, 3], "idx": 0}])
        selector = GoldSelector(table_path="tmp.table", vectorizer=GoldVectorizer())

        with pytest.raises(ValueError):
            selector._check_dataset(dataset)

    def test_check_dataset_missing_idx(self):
        dataset = DummyDataset([{"features": torch.tensor([1.0])}])
        selector = GoldSelector(table_path="tmp.table", vectorizer=GoldVectorizer())

        with pytest.raises(ValueError):
            selector._check_dataset(dataset)

    def test_initialize_table(self):
        table_path = "unit_test.test_select_initialize"

        sample = {"features": torch.rand(5, 2), "idx": 7}
        dataset = DummyDataset([sample])

        selector = GoldSelector(
            table_path=table_path,
            vectorizer=GoldVectorizer(),
            if_exists="replace_force",
        )

        pxt_table = selector._initialize_table(dataset)

        assert pxt_table.count() == 0
        assert set(pxt_table.columns()) == {
            "vector",
            "sample_idx",
            "idx",
            "selected",
            "chunked",
        }

        # Cleanup: drop the created table to avoid polluting environment
        try:
            pxt.drop_table(table_path)
        except Exception:
            # best effort cleanup
            pass

    def test_sequential_store_vectors_in_table(self):
        table_path = "unit_test.test_select_store_vectors"

        # two samples each with a single vector (1xD)
        sample0 = {
            "features": torch.rand(
                5,
                2,
            ),
            "idx": 0,
        }
        sample1 = {
            "features": torch.rand(
                5,
                2,
            ),
            "idx": 1,
        }
        dataset = DummyDataset([sample0, sample1])

        selector = GoldSelector(
            table_path=table_path,
            vectorizer=GoldVectorizer(),
            if_exists="replace_force",
        )

        pxt_table = selector._sequential_store_vectors_in_table(dataset)

        # Expect two vectors inserted (one per sample)
        assert pxt_table.count() == 4

        # Cleanup
        try:
            pxt.drop_table(table_path)
        except Exception:
            pass

    def test_sequential_store_vectors_in_table_with_collate_fn(self):
        table_path = "unit_test.test_select_store_vectors"

        def collate_fn(batch):
            return {
                "features": torch.stack([item["features"] for item in batch]),
                "idx": torch.tensor([item["idx"] for item in batch]),
            }

        # two samples each with a single vector (1xD)
        sample0 = {
            "features": torch.rand(
                5,
                2,
            ),
            "idx": 0,
        }
        sample1 = {
            "features": torch.rand(
                5,
                2,
            ),
            "idx": 1,
        }
        dataset = DummyDataset([sample0, sample1])

        selector = GoldSelector(
            table_path=table_path,
            vectorizer=GoldVectorizer(),
            collate_fn=collate_fn,
            if_exists="replace_force",
        )

        pxt_table = selector._sequential_store_vectors_in_table(dataset)

        # Expect two vectors inserted (one per sample)
        assert pxt_table.count() == 4

        # Cleanup
        try:
            pxt.drop_table(table_path)
        except Exception:
            pass

    def test_sequential_select(self):
        """Test _sequential_select using a real PixelTable table.

        This test will be skipped if the `pixeltable` package is not available.
        We only monkeypatch `_coresubset_selection` to control the selected indices
        while keeping the real table creation and insertion logic.
        """
        table_path = "unit_test.test_select_sequential"

        # prepare two samples, each with a single vector (shape: (1, D))
        dataset = DummyDataset(
            [{"features": torch.rand(5, 2), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            vectorizer=GoldVectorizer(),
            if_exists="replace_force",
        )

        # Run selection (sequential path)
        selected = selector.select(dataset, select_count=10)

        # We expect one sample to be selected (the one corresponding to the first vector chosen by fake_coresubset)
        assert isinstance(selected, set)
        assert len(selected) == 10

        # Cleanup created table
        try:
            pxt.drop_table(table_path)
        except Exception:
            pass

    def test_sequential_select_with_chunk(self):
        """Test _sequential_select using a real PixelTable table.

        This test will be skipped if the `pixeltable` package is not available.
        We only monkeypatch `_coresubset_selection` to control the selected indices
        while keeping the real table creation and insertion logic.
        """
        table_path = "unit_test.test_select_sequential"

        # prepare two samples, each with a single vector (shape: (1, D))
        dataset = DummyDataset(
            [{"features": torch.rand(5, 2), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            vectorizer=GoldVectorizer(),
            if_exists="replace_force",
            chunk=21,
        )

        # Run selection (sequential path)
        selected = selector.select(dataset, select_count=10)

        # We expect one sample to be selected (the one corresponding to the first vector chosen by fake_coresubset)
        assert isinstance(selected, set)
        assert len(selected) == 10

        # Cleanup created table
        try:
            pxt.drop_table(table_path)
        except Exception:
            pass

    def test_sequential_reducer(self):
        """Test _sequential_select using a real PixelTable table.

        This test will be skipped if the `pixeltable` package is not available.
        We only monkeypatch `_coresubset_selection` to control the selected indices
        while keeping the real table creation and insertion logic.
        """
        table_path = "unit_test.test_select_sequential"

        # prepare two samples, each with a single vector (shape: (1, D))
        dataset = DummyDataset(
            [{"features": torch.rand(5, 2), "idx": idx} for idx in range(100)]
        )

        selector = GoldSelector(
            table_path=table_path,
            vectorizer=GoldVectorizer(),
            reducer=GoldReducer(PCA(n_components=3)),
            if_exists="replace_force",
        )

        # Run selection (sequential path)
        selected = selector.select(dataset, select_count=10)

        # We expect one sample to be selected (the one corresponding to the first vector chosen by fake_coresubset)
        assert isinstance(selected, set)
        assert len(selected) == 10

        # Cleanup created table
        try:
            pxt.drop_table(table_path)
        except Exception:
            pass

    def test_select_with_target_key(self):
        """Ensure selection works when features to select are provided under a custom target key.

        The dataset samples include a `target` tensor which should be used by the selector
        when `select_target_key="target"` is passed.
        """
        table_path = "unit_test.test_select_with_target"

        # prepare samples that include both features and a target tensor
        dataset = DummyDataset(
            [
                {
                    "features": torch.rand(5, 2),
                    "target": torch.ones(
                        1,
                        2,
                    ),
                    "idx": idx,
                }
                for idx in range(50)
            ]
        )

        selector = GoldSelector(
            table_path=table_path,
            vectorizer=GoldVectorizer(),
            select_target_key="target",
            if_exists="replace_force",
        )

        selected = selector.select(dataset, select_count=10)

        assert isinstance(selected, set)
        assert len(selected) == 10

        # Cleanup created table
        try:
            pxt.drop_table(table_path)
        except Exception:
            pass
