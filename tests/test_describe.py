import logging
from collections import defaultdict

import numpy as np
import pytest
import torch

import pixeltable as pxt
from pixeltable import Error

from goldener.describe import GoldDescriptor
from goldener.embed import GoldTorchEmbeddingToolConfig, GoldTorchEmbeddingTool
from goldener.torch_utils import collate_keeping_sequences_as_sequences
from goldener.vectorize import GoldTensorVectorizationTool


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
    config = GoldTorchEmbeddingToolConfig(model=model, layers=None)
    return GoldTorchEmbeddingTool(config)


@pytest.fixture
def vectorizer():
    return GoldTensorVectorizationTool()


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
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def test_distribute_cannot_be_enabled(self, embedder):
        error = "Distributed processing is not implemented for GoldDescriptor"
        with pytest.raises(NotImplementedError, match=error):
            GoldDescriptor(
                table_path="unit_test.test_distribute",
                embedder=embedder,
                distribute=True,
            )

        descriptor = GoldDescriptor(
            table_path="unit_test.test_distribute",
            embedder=embedder,
        )
        assert descriptor.distribute is False

        with pytest.raises(NotImplementedError, match=error):
            descriptor.distribute = True
        assert descriptor.distribute is False

    def test_estimation_time_batch_count_must_be_positive(self, embedder):
        with pytest.raises(
            ValueError, match="estimation_time_batch_count must be a positive integer"
        ):
            GoldDescriptor(
                table_path="unit_test.test_estimation_time_batch_count",
                embedder=embedder,
                estimation_time_batch_count=0,
            )

        with pytest.raises(
            ValueError, match="estimation_time_batch_count must be a positive integer"
        ):
            GoldDescriptor(
                table_path="unit_test.test_estimation_time_batch_count",
                embedder=embedder,
                estimation_time_batch_count=-1,
            )

        descriptor = GoldDescriptor(
            table_path="unit_test.test_estimation_time_batch_count",
            embedder=embedder,
            estimation_time_batch_count=3,
        )
        assert descriptor.estimation_time_batch_count == 3

    def test_simple_describe_in_table(self, embedder):
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

    def test_describe_in_table_without_idx(self, embedder):
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

    def test_describe_in_table_with_non_dict_item(self, embedder):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            collate_fn=lambda x: [d["data"] for d in x],
            device=torch.device("cpu"),
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample must be a dictionary"):
            desc.describe_in_table(DummyDataset())

    def test_describe_in_table_with_missing_data_key(self, embedder):
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

    def test_describe_in_table_with_collate_fn(self, embedder):
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

    def test_describe_in_table_with_max_batches(self, embedder):
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

    def test_describe_in_table_with_table_input(self, embedder):
        src_path = "unit_test.src_table_input"
        desc_path = "unit_test.test_describe_from_table"

        source_rows = [
            {"idx": 0, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
            {"idx": 1, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
        ]

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

    def test_describe_from_dataset_with_restrict_to(self, embedder):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe_restrict",
            embedder=embedder,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
        )

        description_table = desc.describe_in_table(
            DummyDataset(dataset_len=6), restrict_to={1, 4}
        )

        assert description_table.count() == 2
        assert {row["idx"] for row in description_table.collect()} == {1, 4}

    def test_describe_from_table_with_restrict_to_from_table(self, embedder):
        src_path = "unit_test.src_table_restrict"
        desc_path = "unit_test.test_describe_restrict_from_table"

        source_rows = [
            {"idx": idx, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"}
            for idx in range(6)
        ]
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        desc = GoldDescriptor(
            table_path=desc_path,
            embedder=embedder,
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
        )

        description_table = desc.describe_in_table(src_table, restrict_to={1, 4})

        assert description_table.count() == 2
        assert {row["idx"] for row in description_table.collect()} == {1, 4}

    def test_describe_in_table_after_restart(self, embedder):
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

    def test_describe_in_table_after_restart_with_restart_disallowed(self, embedder):
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

    def test_describe_in_dataset(self, embedder):
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

    def test_describe_in_dataset_from_table(self, embedder):
        src_path = "unit_test.src_table_input"

        source_rows = [
            {"idx": 0, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
            {"idx": 1, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
        ]

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

    def test_simple_describe_in_table_from_dataset_with_target(
        self, embedder, vectorizer
    ):
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

    def test_simple_describe_in_table_from_dataset_with_target_and_collatefn(
        self, embedder, vectorizer
    ):
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

    def test_describe_in_table_from_table_with_vectorizer(self, embedder, vectorizer):
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

    def test_describe_in_dataset_from_table_with_vectorizer(self, embedder, vectorizer):
        src_path = "unit_test.src_table_input"

        source_rows = [
            {"idx": 0, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
            {"idx": 1, "data": torch.zeros(3, 8, 8).numpy(), "label": "dummy"},
        ]

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

    def test_describe_in_dataset_from_table_with_vectorizer_and_target(
        self, embedder, vectorizer
    ):
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

    def test_describe_in_dataset_from_table_with_excluded_label(
        self, embedder, vectorizer
    ):
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

    def test_describe_in_dataset_from_table_with_vectorizer_and_multitarget(
        self, embedder, vectorizer
    ):
        src_path = "unit_test.src_table_input"

        target = torch.zeros(1, 8, 8).numpy()
        target[0, 0, 0] = 25

        src_table = pxt.create_table(
            src_path,
            schema={
                "idx": pxt.Int,
                "data": pxt.Array[(3, 8, 8), np.float32],
                "label": pxt.Json,
                "target": pxt.Array[(1, 8, 8), np.float32],
            },
            if_exists="replace_force",
        )

        src_table.insert(
            [
                {
                    "idx": 0,
                    "data": torch.zeros(3, 8, 8).numpy(),
                    "label": [
                        "class_1",
                        "class_2",
                    ],
                    "target": target,
                },
                {
                    "idx": 1,
                    "data": torch.zeros(3, 8, 8).numpy(),
                    "label": [
                        "class_1",
                        "class_2",
                    ],
                    "target": target,
                },
            ]
        )

        def collate_fn(batch: list[dict]):
            new_batch_as_list = defaultdict(list)

            for sample in batch:
                new_batch_as_list["idx"].append(sample["idx"])
                new_batch_as_list["label"].append(sample["label"])
                new_batch_as_list["target"].append(sample["target"])
                new_batch_as_list["data"].append(sample["data"])

            return {
                "data": torch.stack(
                    [torch.from_numpy(d) for d in new_batch_as_list["data"]], dim=0
                ),
                "target": torch.stack(
                    [torch.from_numpy(t) for t in new_batch_as_list["target"]], dim=0
                ),
                "idx": new_batch_as_list["idx"],
                "label": new_batch_as_list["label"],
            }

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            label_key="label",
            to_keep_schema={"label": pxt.String},
            target_to_label={(25,): "class_1", (0,): "class_2"},
            batch_size=2,
            collate_fn=collate_fn,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        description = desc.describe_in_table(src_table)

        assert description.count() == 128
        for row in description.collect():
            assert row["idx"] in (0, 1)
            assert row["embeddings"].shape == (4,)
            assert row["label"] in ("class_1", "class_2")

    def test_describe_in_dataset_from_table_with_vectorizer_and_merge_multilabel(
        self, embedder, vectorizer
    ):
        src_path = "unit_test.src_table_input"

        target = torch.zeros(1, 8, 8).numpy()
        target[0, 0, 0] = 25

        src_table = pxt.create_table(
            src_path,
            schema={
                "idx": pxt.Int,
                "data": pxt.Array[(3, 8, 8), np.float32],
                "label": pxt.Json,
                "target": pxt.Array[(1, 8, 8), np.float32],
            },
            if_exists="replace_force",
        )

        src_table.insert(
            [
                {
                    "idx": 0,
                    "data": torch.zeros(3, 8, 8).numpy(),
                    "label": [
                        "class_1",
                        "class_2",
                    ],
                    "target": target,
                },
                {
                    "idx": 1,
                    "data": torch.zeros(3, 8, 8).numpy(),
                    "label": [
                        "class_1",
                        "class_2",
                    ],
                    "target": target,
                },
            ]
        )

        def collate_fn(batch: list[dict]):
            new_batch_as_list = defaultdict(list)

            for sample in batch:
                new_batch_as_list["idx"].append(sample["idx"])
                new_batch_as_list["label"].append(sample["label"])
                new_batch_as_list["target"].append(sample["target"])
                new_batch_as_list["data"].append(sample["data"])

            return {
                "data": torch.stack(
                    [torch.from_numpy(d) for d in new_batch_as_list["data"]], dim=0
                ),
                "target": torch.stack(
                    [torch.from_numpy(t) for t in new_batch_as_list["target"]], dim=0
                ),
                "idx": new_batch_as_list["idx"],
                "label": new_batch_as_list["label"],
            }

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=vectorizer,
            target_to_label={(25,): "class_1", (0,): "class_2"},
            label_key="label",
            merge_multilabels=True,
            to_keep_schema={"label": pxt.String},
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
            assert row["label"] == "class_1_class_2"

    def test_describe_in_dataset_from_table_with_vectorizer_and_multitarget_without_zeros(
        self, embedder, vectorizer
    ):
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

    def test_describe_in_table_with_plain_multilabel_and_default_collate(
        self, embedder, vectorizer
    ):
        """Regression test for #202.

        When collate_fn is left at its default (None), PyTorch's own
        default_collate transposes list-valued batch fields (like a
        multi-label list) across the batch dimension instead of keeping
        one list per sample. Each sample then silently loses all but one
        of its labels.
        """

        class MultiLabelDataset:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "data": torch.zeros(3, 8, 8),
                    "idx": idx,
                    "label": ["class_1", "class_2"],
                }

        desc = GoldDescriptor(
            table_path="unit_test.test_describe_multilabel",
            embedder=embedder,
            vectorizer=vectorizer,
            label_key="label",
            to_keep_schema={"label": pxt.String},
            batch_size=2,
            collate_fn=None,
            device=torch.device("cpu"),
            allow_existing=False,
            drop_table=True,
        )

        description = desc.describe_in_table(MultiLabelDataset())

        labels_by_idx = defaultdict(set)
        for row in description.collect():
            labels_by_idx[row["idx"]].add(row["label"])

        assert labels_by_idx[0] == {"class_1", "class_2"}
        assert labels_by_idx[1] == {"class_1", "class_2"}

    def test_collate_fn_defaults_to_collate_keeping_sequences_as_sequences(
        self, embedder
    ):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
        )
        assert desc.collate_fn is collate_keeping_sequences_as_sequences

        def custom_collate_fn(batch):
            return batch

        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            collate_fn=custom_collate_fn,
        )
        assert desc.collate_fn is custom_collate_fn

    def test_describe_in_table_after_restart_with_vectorizer(self, embedder):
        desc = GoldDescriptor(
            table_path="unit_test.test_describe",
            embedder=embedder,
            vectorizer=GoldTensorVectorizationTool(),
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

    def test_describe_with_force_fix_description_false(self, embedder):
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


class TestEstimateComputationTime:
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def test_returns_estimate_when_dataset_size_known(self, embedder):
        desc = GoldDescriptor(
            table_path="unit_test.describe_estimate",
            embedder=embedder,
            batch_size=2,
        )
        dataset = DummyDataset(dataset_len=10)

        estimate = desc.estimate_computation_time(dataset)

        assert estimate is not None
        assert estimate > 0

    def test_returns_none_when_dataset_size_unknown(self, embedder, caplog):
        from torch.utils.data import IterableDataset

        class NoLenDataset(IterableDataset):
            def __init__(self):
                self.data = [{"data": torch.zeros(3, 8, 8), "idx": i} for i in range(4)]

            def __iter__(self):
                return iter(self.data)

        desc = GoldDescriptor(
            table_path="unit_test.describe_estimate_nolen",
            embedder=embedder,
            batch_size=2,
        )

        with caplog.at_level(logging.INFO):
            estimate = desc.estimate_computation_time(NoLenDataset())

        assert estimate is None
        assert "GoldDescriptor: average computation time per batch" in caplog.text

    def test_returns_none_when_dataset_is_empty(self, embedder):
        desc = GoldDescriptor(
            table_path="unit_test.describe_estimate_empty",
            embedder=embedder,
            batch_size=2,
        )
        dataset = DummyDataset(dataset_len=0)

        estimate = desc.estimate_computation_time(dataset)

        assert estimate is None

    def test_returns_estimate_from_table_input(self, embedder):
        sample = [
            {"data": np.zeros((3, 8, 8), dtype=np.float32), "idx": i} for i in range(4)
        ]
        source_table = pxt.create_table(
            "unit_test.describe_estimate_source",
            source=sample,
            if_exists="replace_force",
        )

        desc = GoldDescriptor(
            table_path="unit_test.describe_estimate_table",
            embedder=embedder,
            batch_size=2,
        )

        estimate = desc.estimate_computation_time(source_table)

        assert estimate is not None
        assert estimate > 0

    def test_estimation_time_batch_count_is_respected(self, embedder):
        class CountingEmbedder:
            def __init__(self, embedder):
                self.embedder = embedder
                self.calls = 0

            def embed_and_fuse(self, data):
                self.calls += 1
                return self.embedder.embed_and_fuse(data)

        counting_embedder = CountingEmbedder(embedder)
        desc = GoldDescriptor(
            table_path="unit_test.describe_estimate_count",
            embedder=counting_embedder,
            batch_size=2,
            estimation_time_batch_count=3,
        )

        estimate = desc.estimate_computation_time(DummyDataset(dataset_len=10))

        assert estimate is not None
        assert counting_embedder.calls == 3

    def test_estimation_does_not_create_or_fill_the_description_table(self, embedder):
        table_path = "unit_test.describe_estimate_no_side_effect"

        desc = GoldDescriptor(
            table_path=table_path,
            embedder=embedder,
            batch_size=2,
        )

        estimate = desc.estimate_computation_time(DummyDataset(dataset_len=4))

        assert estimate is not None
        try:
            table = pxt.get_table(table_path, if_not_exists="ignore")
        except Error:
            table = None
        assert table is None

    def test_returns_estimate_with_vectorizer(self, embedder, vectorizer):
        desc = GoldDescriptor(
            table_path="unit_test.describe_estimate_with_vectorizer",
            embedder=embedder,
            vectorizer=vectorizer,
            batch_size=2,
        )

        estimate = desc.estimate_computation_time(DummyDataset(dataset_len=4))

        assert estimate is not None
        assert estimate > 0
