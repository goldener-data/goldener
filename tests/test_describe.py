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

    def test_with_out_idx(self, extractor):
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


def test_golddescriptor_unwraps_primitive_non_tensors_and_is_searchable(self, extractor):
        # Define a sample object that includes various non-tensor primitive types
        class SampleObject:
            def __init__(self, idx, data_tensor, name, age, is_active, tags, extra_info):
                self.idx = idx
                self.data = data_tensor  # PyTorch tensor
                self.name = name        # string
                self.age = age          # integer
                self.is_active = is_active # boolean
                self.tags = tags        # list (complex non-tensor)
                self.extra_info = extra_info # dict (complex non-tensor)

        # Create a dummy dataset that yields instances of SampleObject
        class SampleObjectDataset:
            def __init__(self):
                self.items = [
                    SampleObject(
                        0,
                        torch.zeros(3, 8, 8),
                        "Alice",
                        30,
                        True,
                        ["tag1", "tag2"],
                        {"city": "NYC", "zip": "10001"}
                    ),
                    SampleObject(
                        1,
                        torch.ones(3, 8, 8),
                        "Bob",
                        25,
                        False,
                        ["tag3"],
                        {"city": "LA", "zip": "90001"}
                    ),
                ]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        table_path = "unit_test.test_describe_unwrapped"
        desc = GoldDescriptor(
            table_path=table_path,
            extractor=extractor,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            if_exists="replace_force",
        )

        table = desc.describe(SampleObjectDataset())

        # Assert table count
        assert table.count() == 2

        # Verify schema for unwrapped fields
        schema = {col.name: col for col in table.schema()}

        assert schema['name'].type == pxt.StringType()
        assert schema['age'].type == pxt.IntType()
        assert schema['is_active'].type == pxt.BoolType()
        # Complex non-tensor types should still be stored as String (JSON serialized)
        assert schema['tags'].type == pxt.StringType()
        assert schema['extra_info'].type == pxt.StringType()
        # The tensor field should be an ArrayType (or equivalent for Pixeltable tensors)
        assert schema['data'].type == pxt.ArrayType(pxt.FloatType()) # Assuming float tensors
        assert schema['features'].type == pxt.ArrayType(pxt.FloatType())


        # Perform SQL-like queries using Pixeltable API to confirm searchability

        # Search by string field
        alice_results = table.where(table.name == 'Alice').collect()
        assert len(alice_results) == 1
        assert alice_results[0]['idx'] == 0
        assert alice_results[0]['name'] == 'Alice'
        assert alice_results[0]['age'] == 30
        assert alice_results[0]['is_active'] is True

        # Search by integer field
        age_results = table.where(table.age < 30).collect()
        assert len(age_results) == 1
        assert age_results[0]['idx'] == 1
        assert age_results[0]['name'] == 'Bob'
        assert age_results[0]['age'] == 25

        # Search by boolean field
        inactive_results = table.where(table.is_active == False).collect()
        assert len(inactive_results) == 1
        assert inactive_results[0]['idx'] == 1
        assert inactive_results[0]['name'] == 'Bob'
        assert inactive_results[0]['is_active'] is False

        # Verify that complex types are still stored as JSON (and thus can't be easily queried as native types)
        # We can fetch them, but not query their internal structure easily without JSON functions.
        bob_row = table.where(table.name == 'Bob').collect()[0]
        import json
        assert json.loads(bob_row['tags']) == ["tag3"]
        assert json.loads(bob_row['extra_info']) == {"city": "LA", "zip": "90001"}

        # Cleanup
        try:
            pxt.drop_table(desc.table_path)
        except Exception:
            pass