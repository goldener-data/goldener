import pytest
import torch

import pixeltable as pxt

from goldener.describe import GoldDescriptor
from goldener.extract import TorchGoldFeatureExtractorConfig, TorchGoldFeatureExtractor


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


class DummyDataset:
    def __init__(self, dataset_len: int = 2):
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return {"data": torch.zeros(3, 8, 8), "idx": idx, "label": "dummy"}


class TestGoldDescriptor:
    def test_simple_describe_in_table(self, extractor):
        pxt.drop_dir("unit_test", force=True)
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            to_keep_schema={"label": pxt.String},
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
        )
        table = desc.describe_in_table(DummyDataset())

        assert table.count() == 2
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i
            assert row["features"].shape == (4, 8, 8)
            assert row["label"] == "dummy"

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_without_idx(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        def collate_fn(batch):
            data = torch.stack([b["data"] for b in batch], dim=0)
            return {"data": data}

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            batch_size=2,
            extractor=extractor,
            collate_fn=collate_fn,
            device=torch.device("cpu"),
            allow_existing=False,
        )

        table = desc.describe_in_table(
            DummyDataset(dataset_len=10),
        )
        assert table.count() == 10
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i
            assert row["features"].shape == (4, 8, 8)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_non_dict_item(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            collate_fn=lambda x: [d["data"] for d in x],
            device=torch.device("cpu"),
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample must be a dictionary"):
            desc.describe_in_table(DummyDataset())

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_missing_data_key(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            collate_fn=lambda x: {"not data": "not_data"},
            device=torch.device("cpu"),
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample is missing expected keys"):
            desc.describe_in_table(
                DummyDataset(),
            )

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_collate_fn(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        def collate_fn(batch):
            data = torch.stack([b["data"] for b in batch], dim=0)
            idxs = [b["idx"] for b in batch]
            labels = [b["label"] for b in batch]
            return {"data": data, "idx": idxs, "label": labels}

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            batch_size=2,
            extractor=extractor,
            collate_fn=collate_fn,
            device=torch.device("cpu"),
            allow_existing=False,
        )

        table = desc.describe_in_table(
            DummyDataset(),
        )
        assert table.count() == 2
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i
            assert row["features"].shape == (4, 8, 8)

        desc_table = pxt.get_table(desc.table_path)
        column_schema = desc_table.get_metadata()["columns"]
        for col_name, col_dict in column_schema.items():
            if col_name == "features":
                assert col_dict["type_"] == "Array[(4, 8, 8), Float]"
            elif col_name == "idx":
                assert col_dict["type_"] == "Int"

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_max_batches(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            max_batches=2,
        )

        table = desc.describe_in_table(DummyDataset(dataset_len=10))

        assert table.count() == 4
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_table_input(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_input"
        desc_path = "unit_test.test_describe_from_table"

        source_rows = [
            {"idx": 0, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
            {"idx": 1, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        desc = GoldDescriptor(
            table_path=desc_path,
            extractor=extractor,
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
        )

        description_table = desc.describe_in_table(src_table)

        assert description_table.count() == 2

        for i, row in enumerate(description_table.collect()):
            assert row["idx"] == i
            assert row["features"].shape == (4, 8, 8)
            assert row["label"] == "dummy"

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_after_restart(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=True,
            max_batches=2,
        )
        dataset = DummyDataset(dataset_len=10)
        description_table = desc.describe_in_table(dataset)

        assert description_table.count() == 4
        for i, row in enumerate(description_table.collect()):
            assert row["idx"] == i
        desc.max_batches = None
        description_table = desc.describe_in_table(dataset)

        assert description_table.count() == 10
        for i, row in enumerate(description_table.collect()):
            assert row["idx"] == i
            assert row["features"].shape == (4, 8, 8)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_after_restart_with_restart_disallowed(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            max_batches=2,
        )
        dataset = DummyDataset(dataset_len=10)
        desc.describe_in_table(dataset)

        with pytest.raises(
            ValueError, match="already exists and allow_existing is set to False"
        ):
            desc.describe_in_table(dataset)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(DummyDataset())

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx"] == i
            assert sample["features"].shape == (4, 8, 8)

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset_from_table(self, extractor):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_input"

        source_rows = [
            {"idx": 0, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
            {"idx": 1, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(src_table)

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx"] == i
            assert sample["features"].shape == (4, 8, 8)

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)
