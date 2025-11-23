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
    def __init__(self, output_shape: tuple[int, ...] = (3, 2, 2), dataset_len: int = 2):
        self.dataset_len = dataset_len
        # produce a fixed tensor
        self.output_shape = output_shape

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return {"data": torch.zeros(3, 8, 8), "idx": idx, "label": "dummy"}


class TestGoldDescriptor:
    def test_simple_describe_in_table(self, extractor):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
        )

        table = desc.describe_in_table(DummyDataset())

        assert table.count() == 2
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i
            assert (row["data"] == torch.zeros(3, 8, 8).numpy()).all()
            assert row["label"] == "dummy"
            assert row["features"].shape == (4, 8, 8)

        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass

    def test_describe_in_table_without_idx(self, extractor):
        def collate_fn(batch):
            data = torch.stack([b["data"] for b in batch], dim=0)
            return {"data": data}

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            batch_size=2,
            extractor=extractor,
            collate_fn=collate_fn,
            device=torch.device("cpu"),
            if_exists="replace_force",
        )

        table = desc.describe_in_table(
            DummyDataset(dataset_len=10),
        )
        assert table.count() == 10
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i

        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass

    def test_describe_in_table_with_non_dict_item(self, extractor):
        # Dataset returning a non-dict should raise
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            collate_fn=lambda x: [d["data"] for d in x],
            device=torch.device("cpu"),
            if_exists="replace_force",
        )
        with pytest.raises(ValueError):
            desc.describe_in_table(DummyDataset())

    def test_describe_in_table_with_missing_data_key(self, extractor):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            collate_fn=lambda x: {"not data": "not_data"},
            device=torch.device("cpu"),
            if_exists="replace_force",
        )
        with pytest.raises(ValueError):
            desc.describe_in_table(
                DummyDataset(),
            )

    def test_describe_in_table_with_collate_fn(self, extractor):
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
            if_exists="replace_force",
        )

        table = desc.describe_in_table(
            DummyDataset(),
        )
        assert table.count() == 2
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i
            assert row["label"] == "dummy"

        desc_table = pxt.get_table(desc.table_path)
        column_schema = desc_table.get_metadata()["columns"]
        for col_name, col_dict in column_schema.items():
            if col_name == "features":
                assert col_dict["type_"] == "Array[(4, 8, 8), Float]"
            elif col_name == "data":
                assert col_dict["type_"] == "Array[(3, 8, 8), Float]"
            elif col_name == "idx":
                assert col_dict["type_"] == "Int"
            elif col_name == "label":
                assert col_dict["type_"] == "String"

        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass

    def test_describe_in_table_with_max_batches(self, extractor):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
            max_batches=2,
        )

        # Dataset with 10 items, batch_size=2 means 5 batches total
        # With max_batches=2, only first 2 batches (4 items) should be processed
        table = desc.describe_in_table(DummyDataset(dataset_len=10))

        assert table.count() == 4
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i

        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass

    def test_describe_in_table_with_table_input(self, extractor):
        src_path = "unit_test.src_table_input"
        desc_path = "unit_test.test_describe_from_table"

        # Create a source table with 2 rows
        source_rows = [
            {"idx": 0, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
            {"idx": 1, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
        ]
        try:
            pxt.create_dir("unit_test", if_exists="ignore")
            src_table = pxt.create_table(
                src_path, source=source_rows, if_exists="replace_force"
            )
        except Exception:
            # If creating the table fails for some reason, ensure cleanup and re-raise
            try:
                pxt.drop_table(src_path)
            except Exception:
                pass
            raise

        desc = GoldDescriptor(
            table_path=desc_path,
            extractor=extractor,
            batch_size=1,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
        )

        description_table = desc.describe_in_table(src_table)

        # The description table must contain the same number of rows and the new 'features' column
        assert description_table.count() == 2

        # Check that features were written and have expected shape
        for i, row in enumerate(description_table.collect()):
            assert row["idx"] == i
            assert row["label"] == "dummy"
            assert "features" in description_table.columns()
            assert tuple(row["features"].shape) == (4, 8, 8)

        try:
            pxt.drop_table(desc_path)
        except Exception:
            pass
        try:
            pxt.drop_table(src_path)
        except Exception:
            pass

    def test_describe_in_table_after_restart(self, extractor):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
            max_batches=2,
        )

        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass

        dataset = DummyDataset(dataset_len=10)
        # Dataset with 10 items, batch_size=2 means 5 batches total
        # With max_batches=2, only first 2 batches (4 items) should be processed
        description_table = desc.describe_in_table(dataset)

        assert description_table.count() == 4
        for i, row in enumerate(description_table.collect()):
            assert row["idx"] == i
        desc.max_batches = None
        description_table = desc.describe_in_table(dataset)

        assert description_table.count() == 10
        for i, row in enumerate(description_table.collect()):
            assert row["idx"] == i
            assert row["label"] == "dummy"
            assert "features" in description_table.columns()
            assert tuple(row["features"].shape) == (4, 8, 8)

        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass

    def test_describe_in_dataset(self, extractor):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(DummyDataset())

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx"] == i
            assert (sample["data"] == torch.zeros(3, 8, 8).numpy()).all()
            assert sample["label"] == "dummy"
            assert sample["features"].shape == (4, 8, 8)

        dataset.keep_cache = False

    def test_describe_in_dataset_from_table(self, extractor):
        src_path = "unit_test.src_table_input"

        # Create a source table with 2 rows
        source_rows = [
            {"idx": 0, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
            {"idx": 1, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
        ]
        try:
            pxt.create_dir("unit_test", if_exists="ignore")
            src_table = pxt.create_table(
                src_path, source=source_rows, if_exists="replace_force"
            )
        except Exception:
            # If creating the table fails for some reason, ensure cleanup and re-raise
            try:
                pxt.drop_table(src_path)
            except Exception:
                pass
            raise

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(src_table)

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx"] == i
            assert (sample["data"] == torch.zeros(3, 8, 8).numpy()).all()
            assert sample["label"] == "dummy"
            assert sample["features"].shape == (4, 8, 8)

        dataset.keep_cache = False

        try:
            pxt.drop_table(desc.src_path)
        except Exception:
            pass
