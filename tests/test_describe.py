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
    def __init__(self, output_shape=(3, 2, 2)):
        # produce a fixed tensor
        self.output_shape = output_shape

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"data": torch.zeros(3, 8, 8), "idx": idx, "label": "dummy"}


class TestGoldDescriptor:
    def test_simple_usage(self, extractor):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
        )

        table = desc.describe(DummyDataset())

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

    def test_without_idx(self, extractor):
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

        table = desc.describe(
            DummyDataset(),
        )
        assert table.count() == 2
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i

        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass

    def test_describe_with_non_dict_item(self, extractor):
        # Dataset returning a non-dict should raise
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            collate_fn=lambda x: [d["data"] for d in x],
            device=torch.device("cpu"),
            if_exists="replace_force",
        )
        with pytest.raises(ValueError):
            desc.describe(DummyDataset())

    def test_describe_missing_data_key(self, extractor):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            extractor=extractor,
            collate_fn=lambda x: {"not data": "not_data"},
            device=torch.device("cpu"),
            if_exists="replace_force",
        )
        with pytest.raises(ValueError):
            desc.describe(
                DummyDataset(),
            )

    def test_with_collate_fn(self, extractor):
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

        table = desc.describe(
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
