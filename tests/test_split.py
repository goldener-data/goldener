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
    def test_basic_split(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        splitted = basic_splitter.split(
            dataset=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
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

        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
            class_key="label",
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

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_duplicated_set_names(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="train", ratio=0.5), GoldSet(name="train", ratio=0.3)]
        with pytest.raises(ValueError):
            GoldSplitter(
                sets=sets,
                descriptor=descriptor,
                selector=selector,
                vectorizer=vectorizer,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_class_key_not_found(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="only", ratio=0.5)]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
            class_key="nonexistent",
        )

        with pytest.raises(ValueError):
            splitter.split(
                DummyDataset(
                    [{"data": torch.rand(3, 8, 8), "idx": 0, "label": "dummy"}]
                )
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_set_with_0_population(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="only", ratio=0.01)]
        splitter = GoldSplitter(
            sets=sets, descriptor=descriptor, selector=selector, vectorizer=vectorizer
        )

        with pytest.raises(ValueError):
            splitter.split(
                dataset=DummyDataset(
                    [
                        {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                        for idx in range(1)
                    ]
                )
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_class_with_0_population(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="only", ratio=0.01)]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
            class_key="label",
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
        assert splitter.selector.max_batches == 1
        assert splitter.vectorizer.max_batches == 1

        splitted = splitter.split(
            dataset=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )

        assert len(splitted) == 2
        # Only 2 items total (1 batch with batch_size=2)
        assert len(splitted["train"]) + len(splitted["val"]) == 2

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)
