from collections import defaultdict

import torch
import pytest
import pixeltable as pxt
from goldener.pxt_utils import pxt_torch_dataset_collate_fn
from goldener.vectorize import (
    GoldTensorVectorizationTool,
    Filter2DWithCount,
    FilterLocation,
    GoldVectorizer,
    Vectorized,
    unwrap_vectors_in_batch,
    vectorize_and_unwrap_in_batch,
)


class TestGoldTensorVectorizationTool:
    def make_tensor(self, shape=(2, 5, 2)):
        return torch.randint(0, 100, shape)

    def test_vectorize_no_y(self):
        x = self.make_tensor()
        v = GoldTensorVectorizationTool()
        vec = v.vectorize(x)
        assert vec.vectors.shape == (4, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 0, 1, 1]))

    def test_vectorize_with_different_channel_pos(self):
        x = self.make_tensor((2, 2, 5))
        v = GoldTensorVectorizationTool(channel_pos=2)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (4, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 0, 1, 1]))

    def test_vectorize_with_y(self):
        x = self.make_tensor()
        y = torch.ones(2, 1, 2)
        y[0, 0, 0] = 0
        v = GoldTensorVectorizationTool()
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (3, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1, 1]))

    def test_vectorize_with_y_and_full_zero(self):
        x = self.make_tensor((2, 2, 5))
        y = torch.zeros(2, 2)
        v = GoldTensorVectorizationTool(channel_pos=2)
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (4, 5)

    def test_vectorize_with_keep(self):
        x = self.make_tensor()
        keep = Filter2DWithCount(
            filter_count=1, filter_location=FilterLocation.START, keep=True
        )
        v = GoldTensorVectorizationTool(keep=keep)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_with_remove(self):
        x = self.make_tensor()
        remove = Filter2DWithCount(
            filter_count=1, filter_location=FilterLocation.END, keep=False
        )
        v = GoldTensorVectorizationTool(remove=remove)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_with_keep_and_remove(self):
        x = self.make_tensor((2, 5, 3))
        keep = Filter2DWithCount(
            filter_count=2, filter_location=FilterLocation.START, keep=True
        )
        remove = Filter2DWithCount(
            filter_count=1, filter_location=FilterLocation.END, keep=False
        )
        v = GoldTensorVectorizationTool(keep=keep, remove=remove)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_with_random(self):
        x = self.make_tensor()
        v = GoldTensorVectorizationTool(
            random=Filter2DWithCount(
                filter_count=1, filter_location=FilterLocation.RANDOM, keep=True
            )
        )
        vec = v.vectorize(x)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_with_input_selection_tool(self):
        class FirstAndLastSelectionTool:
            def __init__(self):
                self.seen = None

            def select(self, x, k, anchors=None):
                self.seen = x.clone()
                assert k == 2
                assert anchors is None
                return [0, len(x) - 1]

        x = torch.arange(15).reshape(1, 3, 5)
        selection_tool = FirstAndLastSelectionTool()
        v = GoldTensorVectorizationTool(
            in_selection_tool=selection_tool,
            in_selection_size=2,
        )

        vec = v.vectorize(x)

        assert selection_tool.seen.shape == (5, 3)
        assert torch.equal(vec.vectors, selection_tool.seen[[0, 4]])
        assert torch.equal(vec.batch_indices, torch.tensor([0, 0]))

    def test_input_selection_runs_after_random_filter(self):
        class RecordingSelectionTool:
            def __init__(self):
                self.input_count = None

            def select(self, x, k, anchors=None):
                self.input_count = len(x)
                return list(range(k))

        selection_tool = RecordingSelectionTool()
        v = GoldTensorVectorizationTool(
            random=Filter2DWithCount(
                filter_count=3,
                filter_location=FilterLocation.RANDOM,
                keep=True,
                generator=torch.Generator().manual_seed(7),
            ),
            in_selection_tool=selection_tool,
            in_selection_size=2,
        )

        vec = v.vectorize(torch.arange(18).reshape(1, 3, 6))

        assert selection_tool.input_count == 3
        assert vec.vectors.shape == (2, 3)

    @pytest.mark.parametrize("size", [0, -1, 0.0, -0.5, 1.5])
    def test_input_selection_size_validation(self, size):
        with pytest.raises(ValueError, match="in_selection_size"):
            GoldTensorVectorizationTool(in_selection_size=size)

    def test_input_selection_accepts_fractional_size(self):
        class RecordingSelectionTool:
            def select(self, x, k, anchors=None):
                assert k == 2
                return list(range(k))

        vectorizer = GoldTensorVectorizationTool(
            in_selection_tool=RecordingSelectionTool(), in_selection_size=0.5
        )

        vec = vectorizer.vectorize(torch.arange(12).reshape(1, 3, 4))

        assert vec.vectors.shape == (2, 3)

    def test_vectorize_with_transform_y(self):
        x = self.make_tensor()
        shape = x.shape
        y = 10 * torch.ones((shape[0], 1, shape[2]))
        y[0, 0, 0] = 3
        y[1, 0, 0] = 3

        def transform_y(y):
            # Only keep rows where y > 5
            return (y > 5).to(torch.int64)

        v = GoldTensorVectorizationTool(transform_y=transform_y)
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_shape_mismatch(self):
        x = self.make_tensor()
        y = torch.ones(2, 1, 3)
        v = GoldTensorVectorizationTool()
        with pytest.raises(ValueError):
            v.vectorize(x, y)

    def test_vectorize_2d_input(self):
        x = self.make_tensor((4, 5))
        v = GoldTensorVectorizationTool()
        with pytest.raises(ValueError):
            v.vectorize(x)

    def test_vectorizer_invalid_keep_type_random(self):
        # keep filter cannot be random
        keep = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.RANDOM,
            keep=True,
        )
        with pytest.raises(ValueError, match="keep"):
            GoldTensorVectorizationTool(keep=keep)

    def test_vectorizer_invalid_keep_type_not_keeping(self):
        # keep filter must have keep=True
        keep = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.START,
            keep=False,
        )
        with pytest.raises(ValueError, match="keep"):
            GoldTensorVectorizationTool(keep=keep)

    def test_vectorizer_invalid_remove_type_random(self):
        # remove filter cannot be random
        remove = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.RANDOM,
            keep=False,
        )
        with pytest.raises(ValueError, match="remove"):
            GoldTensorVectorizationTool(remove=remove)

    def test_vectorizer_invalid_remove_type_not_removing(self):
        # remove filter must have keep=False
        remove = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.START,
            keep=True,
        )
        with pytest.raises(ValueError, match="remove"):
            GoldTensorVectorizationTool(remove=remove)

    def test_vectorizer_invalid_random_type_not_random(self):
        # random filter must be random
        rand = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.START,
            keep=True,
        )
        with pytest.raises(ValueError, match="random"):
            GoldTensorVectorizationTool(random=rand)

    def test_vectorizer_invalid_random_type_not_keeping(self):
        # random filter must have keep=True so it selects indices to keep
        rand = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.RANDOM,
            keep=False,
        )
        with pytest.raises(ValueError, match="random"):
            GoldTensorVectorizationTool(random=rand)

    def test_vectorizer_valid_filters_combination(self):
        # Sanity check: valid combination should construct without errors
        keep = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.START,
            keep=True,
        )
        remove = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.END,
            keep=False,
        )
        rand = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.RANDOM,
            keep=True,
        )
        v = GoldTensorVectorizationTool(keep=keep, remove=remove, random=rand)
        x = self.make_tensor()
        _ = v.vectorize(x)

    def test_vectorize_with_fusion_strategy_average(self):
        x = torch.tensor(
            [
                [
                    [1.0, 3.0, 5.0],
                    [2.0, 4.0, 6.0],
                ],
                [
                    [10.0, 30.0, 50.0],
                    [20.0, 40.0, 60.0],
                ],
            ]
        )

        from goldener.embed import EmbeddingFusionStrategy

        v = GoldTensorVectorizationTool(fusion_strategy=EmbeddingFusionStrategy.AVERAGE)
        vec = v.vectorize(x)

        assert vec.vectors.shape[0] == 2
        assert vec.vectors.shape[1] == 2

        expected_sample0 = torch.tensor([3.0, 4.0])
        expected_sample1 = torch.tensor([30.0, 40.0])

        assert torch.allclose(vec.vectors[0], expected_sample0)
        assert torch.allclose(vec.vectors[1], expected_sample1)

        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))


class TestFilter2DWithCount:
    def make_tensor(self):
        # 5x3 tensor with unique values for easy row checking
        return torch.arange(15).reshape(5, 3)

    def test_filter_start_keep(self):
        x = self.make_tensor()
        f = Filter2DWithCount(
            filter_count=2, filter_location=FilterLocation.START, keep=True
        )
        out = f.filter(x)
        assert out.shape[0] == 2
        assert torch.equal(out, x[:2])

    def test_filter_start_remove(self):
        x = self.make_tensor()
        f = Filter2DWithCount(
            filter_count=2, filter_location=FilterLocation.START, keep=False
        )
        out = f.filter(x)
        assert out.shape[0] == 3
        assert torch.equal(out, x[2:])

    def test_filter_end_keep(self):
        x = self.make_tensor()
        f = Filter2DWithCount(
            filter_count=2, filter_location=FilterLocation.END, keep=True
        )
        out = f.filter(x)
        assert out.shape[0] == 2
        assert torch.equal(out, x[-2:])

    def test_filter_end_remove(self):
        x = self.make_tensor()
        f = Filter2DWithCount(
            filter_count=2, filter_location=FilterLocation.END, keep=False
        )
        out = f.filter(x)
        assert out.shape[0] == 3
        assert torch.equal(out, x[:-2])

    def test_filter_random_keep(self):
        x = self.make_tensor()
        generator = torch.Generator().manual_seed(42)
        f = Filter2DWithCount(
            filter_count=2,
            filter_location=FilterLocation.RANDOM,
            keep=True,
            generator=generator,
        )
        out = f.filter(x)
        assert out.shape[0] == 2
        for row in out:
            assert any(torch.equal(row, r) for r in x)

    def test_filter_random_remove(self):
        x = self.make_tensor()
        generator = torch.Generator().manual_seed(42)
        f = Filter2DWithCount(
            filter_count=2,
            filter_location=FilterLocation.RANDOM,
            keep=False,
            generator=generator,
        )
        out = f.filter(x)
        assert out.shape[0] == 3
        for row in out:
            assert any(torch.equal(row, r) for r in x)

    def test_filter_tensor_dict(self):
        x = self.make_tensor()
        d = {"a": x.clone(), "b": x.clone() + 100}
        f = Filter2DWithCount(
            filter_count=2, filter_location=FilterLocation.START, keep=True
        )
        out = f.filter_tensors(d)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"a", "b"}
        assert torch.equal(out["a"], x[:2])
        assert torch.equal(out["b"], x[:2] + 100)

    def test_filter_random_keep_tensor_dict(self):
        x = self.make_tensor()
        d = {"a": x.clone(), "b": x.clone()}
        generator = torch.Generator().manual_seed(123)
        f = Filter2DWithCount(
            filter_count=2,
            filter_location=FilterLocation.RANDOM,
            keep=True,
            generator=generator,
        )
        out = f.filter_tensors(d)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"a", "b"}
        for tensor in out.values():
            assert tensor.shape[0] == 2
            for row in tensor:
                assert any(torch.equal(row, r) for r in x)

    def test_invalid_filter_count(self):
        with pytest.raises(ValueError):
            Filter2DWithCount(filter_count=0)

    def test_non_2d_input(self):
        x = torch.arange(10)
        f = Filter2DWithCount(filter_count=1)
        with pytest.raises(ValueError):
            f.filter(x)
        d = {"a": torch.arange(10)}
        with pytest.raises(ValueError):
            f.filter_tensors(d)

    def test_inconsistent_batch_size_dict(self):
        x = self.make_tensor()
        d = {"a": x, "b": x[:3]}
        f = Filter2DWithCount(filter_count=2)
        with pytest.raises(ValueError):
            f.filter_tensors(d)

    def test_filter_count_greater_than_rows(self):
        x = self.make_tensor()
        # filter_count > number of rows
        f = Filter2DWithCount(
            filter_count=10, filter_location=FilterLocation.START, keep=True
        )
        out = f.filter(x)
        assert (out == x).all()

    def test_dict_output_keys_and_shapes(self):
        x = self.make_tensor()
        d = {"a": x, "b": x + 1}
        f = Filter2DWithCount(
            filter_count=2, filter_location=FilterLocation.START, keep=True
        )
        out = f.filter_tensors(d)
        assert set(out.keys()) == {"a", "b"}
        assert out["a"].shape == (2, 3)
        assert out["b"].shape == (2, 3)


class DummyDataset:
    def __init__(self, dataset_len: int = 2):
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return {"embeddings": torch.zeros(3, 8, 8), "idx": idx, "label": "dummy"}


class TestGoldVectorizer:
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def test_collate_fn_defaults_to_pxt_torch_dataset_collate_fn(self):
        gv = GoldVectorizer(
            table_path="unit_test.vectorize_default_collate",
            vectorizer=GoldTensorVectorizationTool(),
        )
        assert gv.collate_fn is pxt_torch_dataset_collate_fn

        def custom_collate_fn(batch):
            return batch

        gv = GoldVectorizer(
            table_path="unit_test.vectorize_default_collate",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=custom_collate_fn,
        )
        assert gv.collate_fn is custom_collate_fn

    def test_vectorize_in_table_with_plain_multilabel_and_default_collate(self):
        """Regression test for #202 (GoldVectorizer flavor).

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
                    "embeddings": torch.zeros(4, 3),
                    "idx": idx,
                    "label": ["class_1", "class_2"],
                }

        gv = GoldVectorizer(
            table_path="unit_test.vectorize_multilabel",
            vectorizer=GoldTensorVectorizationTool(),
            data_key="embeddings",
            label_key="label",
            to_keep_schema={"label": pxt.String},
            batch_size=2,
            allow_existing=False,
        )

        out_table = gv.vectorize_in_table(MultiLabelDataset())

        labels_by_idx = defaultdict(set)
        for row in out_table.collect():
            labels_by_idx[row["idx"]].add(row["label"])

        assert labels_by_idx[0] == {"class_1", "class_2"}
        assert labels_by_idx[1] == {"class_1", "class_2"}

    def test_vectorize_in_table_from_dataset(self):
        table_path = "unit_test.vectorize_from_dataset"

        dataset = DummyDataset(dataset_len=2)

        gv = GoldVectorizer(
            table_path=table_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )

        table = gv.vectorize_in_table(dataset)

        # each sample has 2 vectors (first dim), dataset_len=2 => total 4 rows
        assert table.count() == 8 * 8 * 2
        for row in table.collect():
            assert "vectorized" in row
            assert row["vectorized"].shape == (3,)
            assert row["label"] == "dummy"

    def test_vectorize_in_table_from_table(self):
        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        source_rows = [
            {"idx": 0, "embeddings": torch.zeros(4, 3).numpy(), "label": "dummy"},
            {"idx": 1, "embeddings": torch.zeros(4, 3).numpy(), "label": "dummy"},
        ]

        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )

        out_table = gv.vectorize_in_table(src_table)
        assert out_table.count() == 3 * 2
        for row in out_table.collect():
            assert "vectorized" in row
            assert row["vectorized"].shape == (4,)
            assert row["label"] == "dummy"

    def test_vectorize_in_table_with_restrict_to(self):
        gv = GoldVectorizer(
            table_path="unit_test.vectorize_restrict",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )

        out_table = gv.vectorize_in_table(
            DummyDataset(dataset_len=6), restrict_to={1, 4}
        )

        assert out_table.count() > 0
        assert {row["idx"] for row in out_table.collect()} == {1, 4}
        for row in out_table.collect():
            assert row["vectorized"] is not None

    def test_vectorize_in_table_with_restrict_to_from_table(self):
        src_path = "unit_test.src_vectorize_restrict"
        gv = GoldVectorizer(
            table_path="unit_test.vectorize_restrict_from_table",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )

        source_rows = [
            {"idx": idx, "embeddings": torch.zeros(4, 3).numpy(), "label": "dummy"}
            for idx in range(6)
        ]
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        out_table = gv.vectorize_in_table(src_table, restrict_to={1, 4})

        assert out_table.count() > 0
        assert {row["idx"] for row in out_table.collect()} == {1, 4}

    def test_vectorize_in_table_with_target(self):
        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        source_rows = [
            {
                "idx": 0,
                "embeddings": torch.zeros(4, 3).numpy(),
                "label": "dummy",
                "target": torch.ones(1, 3).numpy(),
            },
            {
                "idx": 1,
                "embeddings": torch.zeros(4, 3).numpy(),
                "label": "dummy",
                "target": torch.ones(1, 3).numpy(),
            },
        ]

        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )

        out_table = gv.vectorize_in_table(src_table)
        assert out_table.count() == 3 * 2
        for row in out_table.collect():
            assert "vectorized" in row
            assert row["vectorized"].shape == (4,)
            assert row["label"] == "dummy"

    def test_vectorize_in_table_with_multitarget(self):
        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        target = torch.zeros(2, 3)
        target[:, 0] = torch.tensor([25, 25])
        target = target.numpy()

        source_rows = [
            {
                "idx": 0,
                "embeddings": torch.zeros(4, 3).numpy(),
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
            {
                "idx": 1,
                "embeddings": torch.zeros(4, 3).numpy(),
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
        ]

        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            target_to_label={(0, 0): "class_0", (25, 25): "class_1"},
            label_key="label",
            exclude_full_zero_target=False,
        )

        out_table = gv.vectorize_in_table(src_table)
        assert out_table.count() == 3 * 2
        for row_idx, row in enumerate(out_table.collect()):
            assert row["vectorized"].shape == (4,)
            assert row["label"] == "class_0" or row["label"] == "class_1"

    def test_vectorize_in_table_with_multitarget_and_merge_multilabels(self):
        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        target = torch.zeros(2, 3)
        target[:, 0] = torch.tensor([25, 25])
        target = target.numpy()

        source_rows = [
            {
                "idx": 0,
                "embeddings": torch.zeros(4, 3).numpy(),
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
            {
                "idx": 1,
                "embeddings": torch.zeros(4, 3).numpy(),
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
        ]

        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            merge_multilabels=True,
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            target_to_label={(0, 0): "class_0", (25, 25): "class_1"},
            label_key="label",
            exclude_full_zero_target=False,
        )

        out_table = gv.vectorize_in_table(src_table)
        assert out_table.count() == 3 * 2
        for row_idx, row in enumerate(out_table.collect()):
            assert row["vectorized"].shape == (4,)
            assert row["label"] == "class_0_class_1"

    def test_vectorize_in_table_with_multitarget_and_zero_excluded(self):
        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        embeddings = torch.zeros(4, 3).numpy()
        embeddings[:, 0] = torch.ones((4,))

        target = torch.zeros(2, 3).numpy()
        target[:, 0] = torch.tensor([25, 25])

        source_rows = [
            {
                "idx": 0,
                "embeddings": embeddings,
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
            {
                "idx": 1,
                "embeddings": embeddings,
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
        ]

        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            target_to_label={(0, 0): "class_0", (25, 25): "class_1"},
            label_key="label",
            exclude_full_zero_target=True,
        )

        out_table = gv.vectorize_in_table(src_table)
        assert out_table.count() == 2
        for row_idx, row in enumerate(out_table.collect()):
            assert (row["vectorized"] == torch.ones((4,))).all()
            assert row["label"] == "class_1"

    def test_vectorize_in_table_with_excluded_labels(self):
        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        embeddings = torch.zeros(4, 3).numpy()
        embeddings[:, 0] = torch.ones((4,))

        target = torch.zeros(2, 3).numpy()
        target[:, 0] = torch.tensor([25, 25])

        source_rows = [
            {
                "idx": 0,
                "embeddings": embeddings,
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
            {
                "idx": 1,
                "embeddings": embeddings,
                "label": list({"class_0", "class_1"}),
                "target": target,
            },
        ]

        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            to_keep_schema={"label": pxt.String},
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            target_to_label={(0, 0): "class_0", (25, 25): "class_1"},
            label_key="label",
            exclude_labels={"class_0"},
        )

        out_table = gv.vectorize_in_table(src_table)
        assert out_table.count() == 2
        for row_idx, row in enumerate(out_table.collect()):
            assert (row["vectorized"] == torch.ones((4,))).all()
            assert row["label"] == "class_1"

    def test_vectorize_in_table_without_idx(
        self,
    ):
        def collate_fn(batch):
            data = torch.stack([b["embeddings"] for b in batch], dim=0)
            return {"embeddings": data}

        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=collate_fn,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )

        table = gv.vectorize_in_table(
            DummyDataset(dataset_len=2),
        )
        assert table.count() == 128
        idx_vector = set()
        idx = set()
        for row in table.collect():
            idx.add(row["idx"])
            idx_vector.add(row["idx_vector"])

        assert idx == {0, 1}
        assert idx_vector == set(range(128))

    def test_vectorize_in_table_with_non_dict_item(
        self,
    ):
        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=lambda x: [d["embeddings"] for d in x],
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample must be a dictionary"):
            gv.vectorize_in_table(DummyDataset())

    def test_vectorize_in_table_with_missing_data_key(
        self,
    ):
        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="not_present",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample is missing expected keys"):
            gv.vectorize_in_table(DummyDataset())

    def test_vectorize_in_table_with_max_batches(
        self,
    ):
        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            max_batches=2,
        )
        table = gv.vectorize_in_table(
            DummyDataset(dataset_len=3),
        )

        assert table.count() == 128
        for i, row in enumerate(table.collect()):
            assert row["idx_vector"] == i

    def test_vectorize_in_dataset(self):
        table_path = "unit_test.vectorize_dataset"

        dataset = DummyDataset(dataset_len=2)

        gv = GoldVectorizer(
            table_path=table_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            drop_table=True,
        )

        vectorized_dataset = gv.vectorize_in_dataset(dataset)

        count = 0
        for sample in vectorized_dataset:
            assert sample["vectorized"].shape == (3,)
            count += 1

        assert count == 128

    def test_vectorize_in_dataset_from_table(self):
        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        source_rows = [
            {"idx": 0, "embeddings": torch.zeros(3, 8, 8).numpy()},
            {"idx": 1, "embeddings": torch.zeros(3, 8, 8).numpy()},
        ]
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            drop_table=True,
        )

        vectorized_dataset = gv.vectorize_in_dataset(src_table)

        count = 0
        for sample in vectorized_dataset:
            assert sample["vectorized"].shape == (3,)
            count += 1

        assert count == 128

    def test_vectorize_in_table_after_restart(
        self,
    ):
        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=True,
            max_batches=2,
        )
        dataset = DummyDataset(dataset_len=10)
        vectorized_table = gv.vectorize_in_table(dataset)

        assert vectorized_table.count() == 128
        for i, row in enumerate(vectorized_table.collect()):
            assert row["idx_vector"] == i

        gv.max_batches = None
        vectorized_table = gv.vectorize_in_table(dataset)

        assert vectorized_table.count() == 640
        for i, row in enumerate(vectorized_table.collect()):
            assert row["idx_vector"] == i
            assert row["vectorized"].shape == (3,)

    def test_vectorize_in_table_after_restart_with_restart_disallowed(
        self,
    ):
        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=GoldTensorVectorizationTool(),
            collate_fn=None,
            data_key="embeddings",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
            max_batches=2,
        )
        dataset = DummyDataset(dataset_len=10)
        gv.vectorize_in_table(dataset)

        with pytest.raises(
            ValueError, match="already exists and allow_existing is set to False"
        ):
            gv.vectorize_in_table(dataset)


@pytest.fixture
def vectorized():
    vectors = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    batch_indices = torch.tensor([0, 0, 1, 1])
    return Vectorized(vectors=vectors, batch_indices=batch_indices)


@pytest.fixture
def batch():
    target = torch.zeros(2, 1, 3)
    target[0, 0, 0] = 1
    target[1, 0, 0] = 1
    return {
        "idx": list(range(2)),
        "label": ["a", "b"],
        "data": torch.zeros(2, 3, 3),
        "target": target,
    }


class TestUnwrapVectorsInBatch:
    def test_simple_usage(self, vectorized, batch):
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch)
        vectors = result["vectorized"]
        assert (vectors[0] == torch.tensor([0, 1, 2, 3, 4])).all()
        assert (vectors[1] == torch.tensor([5, 6, 7, 8, 9])).all()
        assert (vectors[2] == torch.tensor([10, 11, 12, 13, 14])).all()
        assert (vectors[3] == torch.tensor([15, 16, 17, 18, 19])).all()
        assert result["idx"] == [0, 0, 1, 1]
        assert result["idx_vector"] == [0, 1, 2, 3]

    def test_to_keep_adds_keys(self, vectorized, batch):
        result = unwrap_vectors_in_batch(
            vectorized, "vectorized", batch, to_keep=["label"]
        )
        assert "label" in result
        assert result["label"] == ["a", "a", "b", "b"]

    def test_custom_vectorized_key(self, vectorized, batch):
        result = unwrap_vectors_in_batch(vectorized, "embeddings", batch)
        assert "embeddings" in result
        assert "vectorized" not in result

    def test_with_start(self, vectorized, batch):
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch, starts=100)
        assert result["idx_vector"] == [100, 101, 102, 103]

    def test_with_idx_as_tensor(self, vectorized, batch):
        batch["idx"] = torch.tensor(batch["idx"])
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch, starts=100)
        assert result["idx_vector"] == [100, 101, 102, 103]


@pytest.fixture
def vectorizer() -> GoldTensorVectorizationTool:
    return GoldTensorVectorizationTool(channel_pos=2)


class TestVectorizeAndUnwrapInBatch:
    def test_simple_usage(self, batch, vectorizer):
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key=None,
        )
        assert set(result.keys()) == {"idx", "vectorized", "idx_vector"}
        assert result["idx"] == [0, 0, 0, 1, 1, 1]
        assert result["idx_vector"] == [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
        vectors = result["vectorized"]
        assert len(vectors) == 6
        for i in range(6):
            assert (vectors[i] == torch.zeros(3)).all()

    def test_with_multilabel(self, batch, vectorizer):
        batch["label"] = [["a", "b"], ["c", "d"]]

        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key=None,
            label_key="label",
            to_keep=["label"],
        )
        assert set(result.keys()) == {"idx", "vectorized", "idx_vector", "label"}
        assert result["idx"] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        assert result["idx_vector"] == list(range(12))
        vectors = result["vectorized"]
        assert len(vectors) == 12
        for i in range(12):
            assert (vectors[i] == torch.zeros(3)).all()
        assert result["label"] == [
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
            "c",
            "c",
            "c",
            "d",
            "d",
            "d",
        ]

    def test_with_start(self, batch, vectorizer):
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key=None,
            starts=100,
        )
        assert result["idx_vector"] == [100, 101, 102, 103, 104, 105]

    def test_with_missing_target_key(self, batch, vectorizer):
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key="missing_key",
        )
        vectors = result["vectorized"]
        assert len(vectors) == 6

    def test_with_existing_target_key(self, batch, vectorizer):
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key="target",
        )
        vectors = result["vectorized"]
        assert len(vectors) == 2
