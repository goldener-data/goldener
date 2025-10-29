import pytest
import torch

import pixeltable as pxt
from pixeltable import Error
from torch.utils.data import Dataset

from goldener.describe import GoldDescriptor
from goldener.extract import TorchGoldFeatureExtractorConfig, TorchGoldFeatureExtractor
from goldener.split import GoldSplitter, GoldSet
from goldener.vectorize import GoldVectorizer
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


@pytest.fixture
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


@pytest.fixture
def descriptor(extractor):
    return GoldDescriptor(
        table_path="unit_test.test_split",
        extractor=extractor,
        batch_size=2,
        collate_fn=None,
        device=torch.device("cpu"),
        if_exists="replace_force",
    )


@pytest.fixture
def selector():
    return GoldSelector(
        table_path="unit_test.selector_split",
        vectorizer=GoldVectorizer(),
        if_exists="replace_force",
    )


@pytest.fixture
def basic_splitter(descriptor, selector):
    sets = [GoldSet(name="train", ratio=0.5), GoldSet(name="val", ratio=0.5)]
    return GoldSplitter(sets=sets, descriptor=descriptor, selector=selector)


class TestGoldSplitter:
    def test_basic_split(self, basic_splitter):
        splitted = basic_splitter.split(
            dataset=DummyDataset(
                [{"data": torch.rand(3, 8, 8), "idx": idx} for idx in range(10)]
            )
        )

        assert len(splitted) == 2
        assert len(splitted["train"]) == 5
        assert len(splitted["val"]) == 5

        for path in (
            basic_splitter.descriptor.table_path,
            basic_splitter.selector.table_path,
        ):
            try:
                pxt.drop_table(path)
            except Exception:
                pass

    def test_split_with_label(self, descriptor, selector):
        sets = [
            GoldSet(name="train", ratio=0.5),
        ]
        splitter = GoldSplitter(
            sets=sets, descriptor=descriptor, selector=selector, class_key="label"
        )

        splitted = splitter.split(
            dataset=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": str(idx % 2)}
                    for idx in range(10)
                ]
            )
        )

        assert set(splitted.keys()) == {"train", "not assigned"}
        assert len(splitted["train"]) == 5
        assert len(splitted["not assigned"]) == 5

        for path in (descriptor.table_path, selector.table_path):
            try:
                pxt.drop_table(path)
            except Exception:
                pass

    def test_duplicated_set_names(self, descriptor, selector):
        sets = [GoldSet(name="train", ratio=0.5), GoldSet(name="train", ratio=0.3)]
        with pytest.raises(ValueError):
            GoldSplitter(sets=sets, descriptor=descriptor, selector=selector)

    def test_class_key_not_found(self, descriptor, selector):
        sets = [GoldSet(name="only", ratio=0.5)]
        splitter = GoldSplitter(
            sets=sets, descriptor=descriptor, selector=selector, class_key="nonexistent"
        )

        with pytest.raises(ValueError):
            splitter.split(DummyDataset([{"data": torch.rand(3, 8, 8), "idx": 0}]))

        for path in (descriptor.table_path, selector.table_path):
            try:
                pxt.drop_table(path)
            except Exception:
                pass

    def test_gold_split_class_existing(self, descriptor, selector):
        sets = [GoldSet(name="only", ratio=0.5)]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
        )

        with pytest.raises(Error):
            splitter.split(
                DummyDataset(
                    [
                        {
                            "data": torch.rand(3, 8, 8),
                            "idx": 0,
                            "gold_split_class": "useless",
                        }
                    ]
                )
            )

        for path in (descriptor.table_path, selector.table_path):
            try:
                pxt.drop_table(path)
            except Exception:
                pass

    def test_set_with_0_population(self, descriptor, selector):
        sets = [GoldSet(name="only", ratio=0.1)]
        splitter = GoldSplitter(sets=sets, descriptor=descriptor, selector=selector)

        with pytest.raises(ValueError):
            splitter.split(
                dataset=DummyDataset(
                    [{"data": torch.rand(3, 8, 8), "idx": idx} for idx in range(1)]
                )
            )

        for path in (descriptor.table_path, selector.table_path):
            try:
                pxt.drop_table(path)
            except Exception:
                pass

    def test_class_with_0_population(self, descriptor, selector):
        sets = [GoldSet(name="only", ratio=0.1)]
        splitter = GoldSplitter(
            sets=sets, descriptor=descriptor, selector=selector, class_key="label"
        )

        with pytest.raises(ValueError):
            splitter.split(
                dataset=DummyDataset(
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

        for path in (descriptor.table_path, selector.table_path):
            try:
                pxt.drop_table(path)
            except Exception:
                pass
