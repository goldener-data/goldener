import pytest
import torch

import pixeltable as pxt

from goldener.describe import GoldDescriptor
from goldener.embed import TorchGoldEmbeddingToolConfig, TorchGoldEmbeddingTool
from goldener.vectorize import TensorVectorizer


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
def embedder():
    model = DummyModel()
    config = TorchGoldEmbeddingToolConfig(model=model, layers=None)
    return TorchGoldEmbeddingTool(config)


@pytest.fixture
def vectorizer():
    return TensorVectorizer()


class DummyDataset:
    def __init__(self, dataset_len: int = 2, add_target: bool = False):
        self.dataset_len = dataset_len
        self.add_target = add_target

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        sample = {"data": torch.zeros(3, 8, 8), "idx": idx, "label": "dummy"}

        if not self.add_target:
            return sample

        return sample | {"target": torch.ones(1, 8, 8)}


class DummyMultiSizeDataset:
    """Dataset that returns items with different spatial sizes."""

    _sizes = [(3, 8, 8), (3, 16, 16)]

    def __len__(self):
        return len(self._sizes)

    def __getitem__(self, idx):
        return {"data": torch.zeros(self._sizes[idx]), "idx": idx, "label": "dummy"}


class TestGoldDescriptor:
    def test_simple_describe_in_table(self, embedder):
        pxt.drop_dir("unit_test", force=True)
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
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
            assert row["embeddings"].shape == (4, 8, 8)
            assert row["label"] == "dummy"

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_without_idx(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        def collate_fn(batch):
            data = torch.stack([b["data"] for b in batch], dim=0)
            return {"data": data}

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            batch_size=2,
            embedder=embedder,
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
            assert row["embeddings"].shape == (4, 8, 8)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_non_dict_item(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            collate_fn=lambda x: [d["data"] for d in x],
            device=torch.device("cpu"),
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample must be a dictionary"):
            desc.describe_in_table(DummyDataset())

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_missing_data_key(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            collate_fn=lambda x: {"not data": "not_data"},
            device=torch.device("cpu"),
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample is missing expected keys"):
            desc.describe_in_table(
                DummyDataset(),
            )

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_collate_fn(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        def collate_fn(batch):
            data = torch.stack([b["data"] for b in batch], dim=0)
            idxs = [b["idx"] for b in batch]
            labels = [b["label"] for b in batch]
            return {"data": data, "idx": idxs, "label": labels}

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            batch_size=2,
            embedder=embedder,
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
            assert row["embeddings"].shape == (4, 8, 8)

        desc_table = pxt.get_table(desc.table_path)
        column_schema = desc_table.get_metadata()["columns"]
        for col_name, col_dict in column_schema.items():
            if col_name == "embeddings":
                assert col_dict["type_"] == "Array[(4, 8, 8), float32]"
            elif col_name == "idx":
                assert col_dict["type_"] == "Required[Int]"

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_with_max_batches(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
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

    def test_describe_in_table_with_table_input(self, embedder):
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
            embedder=embedder,
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
            assert row["embeddings"].shape == (4, 8, 8)
            assert row["label"] == "dummy"

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_after_restart(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
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
            assert row["embeddings"].shape == (4, 8, 8)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_after_restart_with_restart_disallowed(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
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

    def test_describe_in_dataset(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(DummyDataset())

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx"] == i
            assert sample["embeddings"].shape == (4, 8, 8)

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset_from_table(self, embedder):
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
            embedder=embedder,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(src_table)

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx"] == i
            assert sample["embeddings"].shape == (4, 8, 8)

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)

    def test_simple_describe_in_table_from_dataset_with_target(
        self, embedder, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            to_keep_schema={"label": pxt.String},
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
        )
        table = desc.describe_in_table(DummyDataset(add_target=True))

        assert table.count() == 128
        for i, row in enumerate(table.collect()):
            assert row["idx_vector"] == i
            assert row["embeddings"].shape == (4,)
            assert row["label"] == "dummy"

        pxt.drop_dir("unit_test", force=True)

    def test_simple_describe_in_table_from_dataset_with_target_and_collatefn(
        self, embedder, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)

        def collate_fn(batch):
            data = torch.stack([b["data"] for b in batch], dim=0)
            idxs = [b["idx"] for b in batch]
            labels = [b["label"] for b in batch]
            targets = torch.stack([b["target"] for b in batch], dim=0)
            return {"data": data, "idx": idxs, "label": labels, "target": targets}

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            to_keep_schema={"label": pxt.String},
            batch_size=2,
            collate_fn=collate_fn,
            device=torch.device("cpu"),
            allow_existing=False,
        )
        table = desc.describe_in_table(DummyDataset(add_target=True))

        assert table.count() == 128
        for i, row in enumerate(table.collect()):
            assert row["idx_vector"] == i
            assert row["embeddings"].shape == (4,)
            assert row["label"] == "dummy"

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_from_table_with_vectorizer(self, embedder, vectorizer):
        pxt.drop_dir("unit_test", force=True)
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            to_keep_schema={"label": pxt.String},
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
        )
        table = desc.describe_in_table(DummyDataset())

        assert table.count() == 128
        for i, row in enumerate(table.collect()):
            assert row["idx_vector"] == i
            assert row["embeddings"].shape == (4,)
            assert row["label"] == "dummy"

        assert set([row["idx"] for row in table.collect()]) == {0, 1}

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset_from_table_with_vectorizer(self, embedder, vectorizer):
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
            embedder=embedder,
            vectorizer=vectorizer,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(src_table)

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx_vector"] == i
            assert sample["embeddings"].shape == (4,)

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset_from_table_with_vectorizer_and_target(
        self, embedder, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_input"

        source_rows = [
            {
                "idx": 0,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": "dummy",
                "target": torch.ones(1, 8, 8).numpy(),
            },
            {
                "idx": 1,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": "dummy",
                "target": torch.ones(1, 8, 8).numpy(),
            },
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(src_table)

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx_vector"] == i
            assert sample["embeddings"].shape == (4,)

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset_from_table_with_excluded_label(
        self, embedder, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_input"

        source_rows = [
            {
                "idx": 0,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": "dummy",
                "target": torch.ones(1, 8, 8).numpy(),
            },
            {
                "idx": 1,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": "excluded",
                "target": torch.ones(1, 8, 8).numpy(),
            },
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            batch_size=2,
            collate_fn=None,
            label_key="label",
            to_keep_schema={"label": pxt.String},
            exclude_labels={"excluded"},
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        dataset = desc.describe_in_dataset(src_table)

        for i, sample in enumerate(iter(dataset)):
            assert sample["idx_vector"] < 64
            assert sample["label"] == "dummy"

        dataset.keep_cache = False

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset_from_table_with_vectorizer_and_multitarget(
        self, embedder, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_input"

        target = torch.zeros(1, 8, 8).numpy()
        target[0, 0, 0] = 25

        source_rows = [
            {
                "idx": 0,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": [
                    "class_1",
                ],
                "target": target,
            },
            {
                "idx": 1,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": [
                    "class_1",
                ],
                "target": target,
            },
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            target_to_label={(25,): "class_1", (0,): "class_2"},
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        description = desc.describe_in_table(src_table)

        assert description.count() == 128
        for row in description.collect():
            assert row["idx"] in (0, 1)
            assert row["embeddings"].shape == (4,)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_dataset_from_table_with_vectorizer_and_multitarget_without_zeros(
        self, embedder, vectorizer
    ):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_input"

        target = torch.zeros(1, 8, 8).numpy()
        target[0, 0, 0] = 25

        source_rows = [
            {
                "idx": 0,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": [
                    "class_1",
                ],
                "target": target,
            },
            {
                "idx": 1,
                "data": torch.zeros(3, 8, 8).numpy(),
                "label": [
                    "class_1",
                ],
                "target": target,
            },
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            target_to_label={(25,): "class_1", (0,): "class_2"},
            exclude_full_zero_target=True,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        description = desc.describe_in_table(src_table)

        assert description.count() == 2
        for row in description.collect():
            assert row["idx"] in (0, 1)
            assert row["embeddings"].shape == (4,)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_in_table_after_restart_with_vectorizer(self, embedder):
        pxt.drop_dir("unit_test", force=True)

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=TensorVectorizer(),
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=True,
            max_batches=2,
        )
        dataset = DummyDataset(dataset_len=10)
        description_table = desc.describe_in_table(dataset)

        assert description_table.count() == 4 * 8 * 8
        for i, row in enumerate(description_table.collect()):
            assert row["idx_vector"] == i
        desc.max_batches = None
        description_table = desc.describe_in_table(dataset)

        assert description_table.count() == 10 * 8 * 8
        for i, row in enumerate(description_table.collect()):
            assert row["idx_vector"] == i
            assert row["embeddings"].shape == (4,)

        pxt.drop_dir("unit_test", force=True)

    def test_describe_with_force_fix_description_false(self, embedder):
        pxt.drop_dir("unit_test", force=True)
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            force_fix_description=False,
            batch_size=1,
            device=torch.device("cpu"),
            allow_existing=False,
        )
        table = desc.describe_in_table(DummyMultiSizeDataset())

        assert table.count() == 2
        rows = table.collect()
        assert rows[0]["embeddings"].shape != rows[1]["embeddings"].shape

        pxt.drop_dir("unit_test", force=True)
