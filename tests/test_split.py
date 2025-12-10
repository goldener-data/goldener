import numpy as np
import pytest
import torch

import pixeltable as pxt
from torch.utils.data import Dataset

from goldener.describe import GoldDescriptor
from goldener.extract import TorchGoldFeatureExtractorConfig, TorchGoldFeatureExtractor
from goldener.pxt_utils import pxt_torch_dataset_collate_fn
from goldener.split import GoldSplitter, GoldSet
from goldener.vectorize import TensorVectorizer, GoldVectorizer
from goldener.select import GoldSelector


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


@pytest.fixture(scope="function")
def extractor():
    model = DummyModel()
    config = TorchGoldFeatureExtractorConfig(model=model, layers=None)
    return TorchGoldFeatureExtractor(config)


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


@pytest.fixture(scope="function")
def descriptor(extractor):
    return GoldDescriptor(
        table_path="unit_test.descriptor_split",
        extractor=extractor,
        to_keep_schema={"label": pxt.String},
        batch_size=2,
        collate_fn=None,
        device=torch.device("cpu"),
        allow_existing=False,
    )


@pytest.fixture(scope="function")
def selector():
    return GoldSelector(
        table_path="unit_test.selector_split",
        to_keep_schema={"label": pxt.String},
        vectorized_key="vectorized",
        batch_size=2,
    )


@pytest.fixture(scope="function")
def vectorizer():
    return GoldVectorizer(
        table_path="unit_test.vectorizer_split",
        vectorizer=TensorVectorizer(),
        to_keep_schema={"label": pxt.String},
        collate_fn=pxt_torch_dataset_collate_fn,
        batch_size=2,
    )


@pytest.fixture(scope="function")
def basic_splitter(descriptor, vectorizer, selector):
    sets = [GoldSet(name="train", ratio=0.5), GoldSet(name="val", ratio=0.5)]
    return GoldSplitter(
        sets=sets, descriptor=descriptor, selector=selector, vectorizer=vectorizer
    )


class TestGoldSplitter:
    def test_split_in_table_from_dataset(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        split_table = basic_splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )

        splitted = basic_splitter.get_split_indices(
            split_table,
            selection_key=basic_splitter.selector.selection_key,
            idx_key="idx_sample",
        )

        assert len(splitted) == 2
        assert len(splitted["train"]) == 5
        assert len(splitted["val"]) == 5

        pxt.get_table(basic_splitter.descriptor.table_path)
        pxt.get_table(basic_splitter.vectorizer.table_path)
        pxt.get_table(basic_splitter.selector.table_path)

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_in_table_from_table(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.test_split_in_table"
        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path,
            source=[
                {
                    "data": torch.rand(3, 8, 8).numpy().astype(np.float32),
                    "idx": idx,
                    "label": "dummy",
                }
                for idx in range(10)
            ],
            if_exists="replace_force",
        )

        split_table = basic_splitter.split_in_table(to_split=src_table)

        splitted = basic_splitter.get_split_indices(
            split_table,
            selection_key=basic_splitter.selector.selection_key,
            idx_key="idx_sample",
        )

        assert len(splitted) == 2
        assert len(splitted["train"]) == 5
        assert len(splitted["val"]) == 5

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_with_label(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [
            GoldSet(name="train", ratio=0.5),
        ]
        selector.class_key = "label"
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
        )

        split_table = splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": str(idx % 2)}
                    for idx in range(10)
                ]
            )
        )
        splitted = splitter.get_split_indices(
            split_table,
            selection_key=splitter.selector.selection_key,
            idx_key="idx_sample",
        )

        assert set(splitted.keys()) == {"train"}
        assert len(splitted["train"]) == 5

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_duplicated_set_names(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="train", ratio=0.5), GoldSet(name="train", ratio=0.3)]
        with pytest.raises(ValueError, match="Set names must be unique"):
            GoldSplitter(
                sets=sets,
                descriptor=descriptor,
                selector=selector,
                vectorizer=vectorizer,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_class_key_not_found(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)
        selector.class_key = "nonexistent"
        sets = [GoldSet(name="only", ratio=0.5)]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
        )

        with pytest.raises(
            ValueError, match="class_key and class_value must be set together"
        ):
            splitter.split_in_table(
                to_split=DummyDataset(
                    [
                        {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                        for idx in range(10)
                    ]
                )
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_set_with_0_population(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="only", ratio=0.01)]
        splitter = GoldSplitter(
            sets=sets, descriptor=descriptor, selector=selector, vectorizer=vectorizer
        )

        with pytest.raises(ValueError, match="in zero samples for dataset of size"):
            splitter.split_in_table(
                to_split=DummyDataset(
                    [
                        {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                        for idx in range(1)
                    ]
                )
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_class_with_0_population(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)
        selector.class_key = "label"
        sets = [GoldSet(name="only", ratio=0.01)]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
        )

        with pytest.raises(ValueError, match="in zero samples for dataset of size"):
            splitter.split_in_table(
                to_split=DummyDataset(
                    [
                        {
                            "data": torch.rand(3, 8, 8),
                            "idx": idx,
                            "label": "A" if idx < 5 else "B",
                        }
                        for idx in range(10)
                    ]
                )
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_max_batches(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="train", ratio=0.5), GoldSet(name="val", ratio=0.5)]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
            max_batches=1,
        )

        assert splitter.descriptor.max_batches == 1

        split_table = splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )
        splitted = splitter.get_split_indices(
            split_table,
            selection_key=splitter.selector.selection_key,
            idx_key="idx_sample",
        )

        assert len(splitted) == 2
        # Only 2 items total (1 batch with batch_size=2)
        assert len(splitted["train"]) + len(splitted["val"]) == 2

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_with_no_remaining_indices(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="train", ratio=0.5), GoldSet(name="val", ratio=0.5)]
        selector.max_batches = 1
        vectorizer.max_batches = 1
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
            max_batches=1,
        )

        with pytest.raises(ValueError, match="in zero samples for dataset of size"):
            splitter.split_in_table(
                to_split=DummyDataset(
                    [
                        {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                        for idx in range(10)
                    ]
                )
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_in_dataset_from_dataset(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)
        basic_splitter.in_described_table = True
        split_dataset = basic_splitter.split_in_dataset(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )

        train_count = 0
        val_count = 0
        for item in split_dataset:
            if item["selected"] == "train":
                train_count += 1
            elif item["selected"] == "val":
                val_count += 1
            else:
                raise ValueError("Unknown split name found in dataset.")

        assert train_count == 5
        assert val_count == 5

        split_dataset.keep_cache = False

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_in_table_from_dataset_with_drop_table(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        basic_splitter.drop_table = True
        basic_splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )

        with pytest.raises(pxt.Error):
            pxt.get_table(basic_splitter.descriptor.table_path)

        with pytest.raises(pxt.Error):
            pxt.get_table(basic_splitter.vectorizer.table_path)

        basic_splitter.in_described_table = True
        basic_splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )

        with pytest.raises(pxt.Error):
            pxt.get_table(basic_splitter.selector.table_path)

        with pytest.raises(pxt.Error):
            pxt.get_table(basic_splitter.vectorizer.table_path)

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_in_table_from_dataset_with_restart(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        dataset = DummyDataset(
            [
                {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                for idx in range(10)
            ]
        )
        basic_splitter.max_batches = 2
        basic_splitter.split_in_table(to_split=dataset)

        basic_splitter.max_batches = None
        split_table = basic_splitter.split_in_table(to_split=dataset)
        splitted = basic_splitter.get_split_indices(
            split_table,
            selection_key=basic_splitter.selector.selection_key,
            idx_key="idx_sample",
        )

        assert len(splitted) == 2
        assert len(splitted["train"]) == 5
        assert len(splitted["val"]) == 5

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_get_split_indices_from_table(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.test_split_in_table"
        pxt.create_dir("unit_test", if_exists="ignore")
        split_table = pxt.create_table(
            src_path,
            source=[
                {
                    "data": torch.rand(3, 8, 8).numpy().astype(np.float32),
                    "idx": idx,
                    "label": "dummy",
                    "set": "train" if idx < 5 else "val",
                }
                for idx in range(10)
            ],
            if_exists="replace_force",
        )

        splitted = GoldSplitter.get_split_indices(
            split_table,
            selection_key="set",
            idx_key="idx",
        )

        for set_name, indices in splitted.items():
            assert len(indices) == 5
            assert set_name in ["train", "val"]

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_get_split_indices_from_dataset(self):
        pxt.drop_dir("unit_test", force=True)

        split_dataset = DummyDataset(
            [
                {
                    "data": torch.rand(3, 8, 8),
                    "idx": idx,
                    "set": "train" if idx < 5 else "val",
                    "label": "dummy",
                }
                for idx in range(10)
            ]
        )

        splitted = GoldSplitter.get_split_indices(
            split_dataset,
            selection_key="set",
            idx_key="idx",
        )

        for set_name, indices in splitted.items():
            assert len(indices) == 5
            assert set_name in ["train", "val"]

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)
