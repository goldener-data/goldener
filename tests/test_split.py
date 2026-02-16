import numpy as np
import pytest
import torch

import pixeltable as pxt
from torch.utils.data import Dataset

from goldener.clusterize import GoldClusterizer, GoldRandomClusteringTool
from goldener.describe import GoldDescriptor
from goldener.extract import TorchGoldFeatureExtractorConfig, TorchGoldFeatureExtractor
from goldener.pxt_utils import pxt_torch_dataset_collate_fn
from goldener.split import GoldSplitter, GoldSet, check_sets_validity
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
        return self._samples[idx]


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
    sets = [GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)]
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
            idx_key="idx",
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
            idx_key="idx",
        )

        assert len(splitted) == 2
        assert len(splitted["train"]) == 5
        assert len(splitted["val"]) == 5

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_with_label(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [
            GoldSet(name="train", size=0.5),
            GoldSet(name="val", size=0.5),
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
            idx_key="idx",
        )

        assert set(splitted.keys()) == {"train", "val"}
        assert len(splitted["train"]) == 5
        assert len(splitted["val"]) == 5

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_duplicated_set_names(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="train", size=0.5), GoldSet(name="train", size=0.5)]
        with pytest.raises(ValueError, match="Set names must be unique"):
            GoldSplitter(
                sets=sets,
                descriptor=descriptor,
                selector=selector,
                vectorizer=vectorizer,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_not_complete_set_sizes_as_float(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        with pytest.raises(
            ValueError, match="Sampling size as float must be equal to 1.0"
        ):
            GoldSplitter(
                sets=[GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.4)],
                descriptor=descriptor,
                selector=selector,
                vectorizer=vectorizer,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_not_complete_set_sizes_as_int(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        gold_splitter = GoldSplitter(
            sets=[GoldSet(name="train", size=2), GoldSet(name="val", size=2)],
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
        )

        with pytest.raises(
            ValueError,
            match="Sampling size as int must be equal to the total number of samples",
        ):
            gold_splitter.split_in_table(
                to_split=DummyDataset(
                    [
                        {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                        for idx in range(10)
                    ]
                )
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_different_type_set_size(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        with pytest.raises(ValueError, match="All set sizes must be of the same type"):
            GoldSplitter(
                sets=[GoldSet(name="train", size=0.5), GoldSet(name="val", size=3)],
                descriptor=descriptor,
                selector=selector,
                vectorizer=vectorizer,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_class_key_not_found(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)
        selector.class_key = "nonexistent"
        sets = [
            GoldSet(name="train", size=0.5),
            GoldSet(name="val", size=0.5),
        ]
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

    def test_set_with_small_population(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [
            GoldSet(name="train", size=0.001),
            GoldSet(name="val", size=0.999),
        ]
        splitter = GoldSplitter(
            sets=sets, descriptor=descriptor, selector=selector, vectorizer=vectorizer
        )

        split_table = splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(2)
                ]
            )
        )

        splitted = splitter.get_split_indices(
            split_table,
            selection_key=splitter.selector.selection_key,
            idx_key="idx",
        )

        assert set(splitted.keys()) == {"train", "val"}
        assert len(splitted["train"]) == 1

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_class_with_small_population_failure(
        self, descriptor, selector, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)
        selector.class_key = "label"
        sets = [
            GoldSet(name="train", size=0.0001),
            GoldSet(name="val", size=0.9999),
        ]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
        )

        with pytest.raises(ValueError, match="which results in zero samples"):
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

        sets = [GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)]
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
            idx_key="idx",
        )

        assert len(splitted) == 2
        # Only 2 items total (1 batch with batch_size=2)
        assert len(splitted["train"]) + len(splitted["val"]) == 2

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_with_no_remaining_indices(self, descriptor, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        sets = [GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)]
        selector.max_batches = 1
        vectorizer.max_batches = 1
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=vectorizer,
            max_batches=1,
        )

        with pytest.raises(
            ValueError,
            match="Not enough data to split among",
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

    def test_split_in_dataset_from_dataset(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)
        basic_splitter.in_described_table = True
        basic_splitter.drop_table = True
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

        assert (
            pxt.get_table(basic_splitter.descriptor.table_path, if_not_exists="ignore")
            is None
        )
        assert (
            pxt.get_table(basic_splitter.vectorizer.table_path, if_not_exists="ignore")
            is None
        )
        assert (
            pxt.get_table(basic_splitter.selector.table_path, if_not_exists="ignore")
            is None
        )

        split_dataset.keep_cache = False

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_in_dataset_without_drop(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)
        basic_splitter.in_described_table = True
        basic_splitter.drop_table = False
        split_dataset = basic_splitter.split_in_dataset(
            to_split=DummyDataset(
                [
                    {"data": torch.rand(3, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )

        assert (
            pxt.get_table(basic_splitter.descriptor.table_path, if_not_exists="ignore")
            is not None
        )
        assert (
            pxt.get_table(basic_splitter.vectorizer.table_path, if_not_exists="ignore")
            is not None
        )
        assert (
            pxt.get_table(basic_splitter.selector.table_path, if_not_exists="ignore")
            is not None
        )

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
            idx_key="idx",
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

    def test_with_descriptor_including_vectorizer(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        descriptor = GoldDescriptor(
            table_path="unit_test.descriptor_split",
            extractor=extractor,
            vectorizer=TensorVectorizer(),
            description_key="vectorized",
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
        )

        selector = GoldSelector(
            table_path="unit_test.selector_split",
            vectorized_key="vectorized",
            batch_size=2,
        )

        sets = [GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)]
        splitter = GoldSplitter(
            sets=sets,
            descriptor=descriptor,
            selector=selector,
            vectorizer=None,
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
            idx_key="idx",
        )

        assert len(splitted) == 2
        # Only 2 items total (1 batch with batch_size=2)
        assert len(splitted["train"]) + len(splitted["val"]) == 2

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_without_descriptor(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        basic_splitter.descriptor = None
        basic_splitter.vectorizer.collate_fn = None
        split_table = basic_splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {"features": torch.rand(4, 8, 8), "idx": idx, "label": "dummy"}
                    for idx in range(10)
                ]
            )
        )
        splitted = basic_splitter.get_split_indices(
            split_table,
            selection_key=basic_splitter.selector.selection_key,
            idx_key="idx",
        )

        assert len(splitted) == 2
        # Only 2 items total (1 batch with batch_size=2)
        assert len(splitted["train"]) + len(splitted["val"]) == 10

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_without_descriptor_nor_vectorizer(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        basic_splitter.descriptor = None
        basic_splitter.vectorizer = None
        split_table = basic_splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {
                        "vectorized": torch.rand(
                            4,
                        ),
                        "idx": idx,
                        "label": "dummy",
                    }
                    for idx in range(10)
                ]
            )
        )
        splitted = basic_splitter.get_split_indices(
            split_table,
            selection_key=basic_splitter.selector.selection_key,
            idx_key="idx",
        )

        assert len(splitted) == 2
        # Only 2 items total (1 batch with batch_size=2)
        assert len(splitted["train"]) + len(splitted["val"]) == 10

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_with_int_sizes(self, basic_splitter):
        pxt.drop_dir("unit_test", force=True)

        basic_splitter.descriptor = None
        basic_splitter.vectorizer = None
        basic_splitter.sets = [
            GoldSet(name="train", size=6),
            GoldSet(name="val", size=4),
        ]
        split_table = basic_splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {
                        "vectorized": torch.rand(
                            4,
                        ),
                        "idx": idx,
                        "label": "dummy",
                    }
                    for idx in range(10)
                ]
            )
        )
        splitted = basic_splitter.get_split_indices(
            split_table,
            selection_key=basic_splitter.selector.selection_key,
            idx_key="idx",
        )

        assert len(splitted) == 2
        assert len(splitted["train"]) + len(splitted["val"]) == 10

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_with_no_descriptor_but_in_described_table(self, selector):
        pxt.drop_dir("unit_test", force=True)

        with pytest.raises(
            ValueError,
            match="in_described_table is set to True, but no descriptor is provided",
        ):
            GoldSplitter(
                sets=[GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)],
                descriptor=None,
                selector=selector,
                vectorizer=None,
                in_described_table=True,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_with_vectorizer_but_wrong_selection_key(self, selector, vectorizer):
        pxt.drop_dir("unit_test", force=True)

        vectorizer.vectorized_key = "wrong_key"
        with pytest.raises(
            ValueError, match="does not match selector's vectorized_key"
        ):
            GoldSplitter(
                sets=[GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)],
                descriptor=None,
                selector=selector,
                vectorizer=vectorizer,
                in_described_table=False,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_with_descriptor_but_wrong_selection_key(self, selector, descriptor):
        pxt.drop_dir("unit_test", force=True)

        descriptor.description_key = "wrong_key"
        with pytest.raises(
            ValueError, match="does not match selector's vectorized_key"
        ):
            GoldSplitter(
                sets=[GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)],
                descriptor=descriptor,
                selector=selector,
                vectorizer=None,
                in_described_table=False,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_with_descriptor_and_vectorizer_but_wrong_key(
        self, selector, descriptor, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)

        descriptor.description_key = "wrong_key"
        with pytest.raises(ValueError, match="does not match vectorizer's data_key"):
            GoldSplitter(
                sets=[GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)],
                descriptor=descriptor,
                selector=selector,
                vectorizer=vectorizer,
                in_described_table=False,
            )

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)

    def test_split_with_clusterizer(self, descriptor, vectorizer, selector):
        pxt.drop_dir("unit_test", force=True)

        # Build a simple vectorized table first, then let the splitter handle clustering
        sets = [GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)]

        clusterizer = GoldClusterizer(
            table_path="unit_test.clusterizer_split",
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            class_key="label",
            allow_existing=False,
        )

        splitter = GoldSplitter(
            sets=sets,
            descriptor=None,
            selector=selector,
            vectorizer=None,
            clusterizer=clusterizer,
            n_clusters=2,
        )

        split_table = splitter.split_in_table(
            to_split=DummyDataset(
                [
                    {
                        "vectorized": torch.rand(
                            3,
                        ),
                        "idx": idx_vector // 4,
                        "label": "dummy",
                        "idx_vector": idx_vector,
                    }
                    for idx_vector in range(120)
                ]
            )
        )

        splitted = splitter.get_split_indices(
            split_table,
            selection_key=splitter.selector.selection_key,
            idx_key="idx",
        )

        assert set(splitted.keys()) == {"train", "val"}
        assert len(splitted["train"]) == 15
        assert len(splitted["val"]) == 15

        # Check that the clusterizer table was created and has clusters assigned
        cluster_table = pxt.get_table(clusterizer.table_path)
        assert cluster_table.count() > 0
        distinct_clusters = [
            row[clusterizer.cluster_key]
            for row in cluster_table.select(cluster_table[clusterizer.cluster_key])
            .distinct()
            .collect()
        ]
        # Since we requested 2 clusters, resulting cluster ids should be a subset of {0,1}
        assert set(distinct_clusters).issubset({0, 1})

        pxt.drop_dir("unit_test", if_not_exists="ignore", force=True)


class TestCheckSetsValidity:
    def test_sum_of_sizes_less_than_one(self):
        sets = [GoldSet(name="train", size=0.4), GoldSet(name="val", size=0.5)]
        check_sets_validity(sets, force_max=False)

    def test_sum_of_sizes_equal_one(self):
        sets = [GoldSet(name="train", size=0.5), GoldSet(name="val", size=0.5)]
        check_sets_validity(sets, force_max=True)

    def test_with_float_failure(self):
        with pytest.raises(
            ValueError,
            match="Sampling size as float must be greater than 0.0 and at most 1.0",
        ):
            sets = [GoldSet(name="train", size=0.51), GoldSet(name="val", size=0.5)]
            check_sets_validity(sets, force_max=False)

        with pytest.raises(
            ValueError,
            match="Sampling size as float must be equal to 1.0",
        ):
            sets = [GoldSet(name="train", size=0.49), GoldSet(name="val", size=0.5)]
            check_sets_validity(sets, force_max=True)

    def test_sum_of_sizes_less_than_total(self):
        sets = [GoldSet(name="train", size=2), GoldSet(name="val", size=2)]
        check_sets_validity(sets, total=5, force_max=False)

    def test_sum_of_sizes_equal_total(self):
        sets = [GoldSet(name="train", size=2), GoldSet(name="val", size=3)]
        check_sets_validity(sets, total=5, force_max=True)

    def test_with_int_failure(self):
        with pytest.raises(
            ValueError,
            match="Sampling size as int must be equal to the total number of samples",
        ):
            sets = [GoldSet(name="train", size=3), GoldSet(name="val", size=1)]
            check_sets_validity(sets, total=5, force_max=True)

        with pytest.raises(
            ValueError,
            match="Sampling size as int must be greater than 0 and less or equal than the total number of samples",
        ):
            sets = [GoldSet(name="train", size=3), GoldSet(name="val", size=3)]
            check_sets_validity(sets, total=5, force_max=False)

    def test_same_name(self):
        sets = [GoldSet(name="train", size=3), GoldSet(name="train", size=2)]
        with pytest.raises(ValueError, match="Set names must be unique"):
            check_sets_validity(sets, total=5)
