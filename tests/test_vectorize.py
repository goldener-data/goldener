import torch
import pytest
import pixeltable as pxt
from goldener.vectorize import (
    TensorVectorizer,
    Filter2DWithCount,
    FilterLocation,
    GoldVectorizer,
)


class TestTensorsVectorizer:
    def make_tensor(self, shape=(2, 5, 2)):
        return torch.randint(0, 100, shape)

    def test_vectorize_no_y(self):
        x = self.make_tensor()
        v = TensorVectorizer()
        vec = v.vectorize(x)
        assert vec.vectors.shape == (4, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 0, 1, 1]))

    def test_vectorize_with_different_channel_pos(self):
        x = self.make_tensor((2, 2, 5))
        v = TensorVectorizer(channel_pos=2)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (4, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 0, 1, 1]))

    def test_vectorize_with_y(self):
        x = self.make_tensor()
        y = torch.ones(2, 1, 2)
        y[0, 0, 0] = 0
        v = TensorVectorizer()
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (3, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1, 1]))

    def test_vectorize_with_keep(self):
        x = self.make_tensor()
        keep = Filter2DWithCount(
            filter_count=1, filter_location=FilterLocation.START, keep=True
        )
        v = TensorVectorizer(keep=keep)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_with_remove(self):
        x = self.make_tensor()
        remove = Filter2DWithCount(
            filter_count=1, filter_location=FilterLocation.END, keep=False
        )
        v = TensorVectorizer(remove=remove)
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
        v = TensorVectorizer(keep=keep, remove=remove)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_with_transform_y(self):
        x = self.make_tensor()
        shape = x.shape
        y = 10 * torch.ones((shape[0], 1, shape[2]))
        y[0, 0, 0] = 3
        y[1, 0, 0] = 3

        def transform_y(y):
            # Only keep rows where y > 5
            return (y > 5).to(torch.int64)

        v = TensorVectorizer(transform_y=transform_y)
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_shape_mismatch(self):
        x = self.make_tensor()
        y = torch.ones(2, 1, 3)
        v = TensorVectorizer()
        with pytest.raises(ValueError):
            v.vectorize(x, y)

    def test_vectorize_2d_input(self):
        x = self.make_tensor((4, 5))
        v = TensorVectorizer()
        with pytest.raises(ValueError):
            v.vectorize(x)

    def test_vectorizer_invalid_keep_type(self):
        with pytest.raises(ValueError):
            TensorVectorizer(keep=Filter2DWithCount())

    def test_vectorizer_invalid_remove_type(self):
        with pytest.raises(ValueError):
            TensorVectorizer(remove=Filter2DWithCount())

    def test_vectorizer_invalid_random_type(self):
        with pytest.raises(ValueError):
            TensorVectorizer(
                random_filter=Filter2DWithCount(filter_location=FilterLocation.START)
            )


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
        return {"features": torch.zeros(3, 8, 8), "idx": idx, "label": "dummy"}


class TestGoldVectorizer:
    def test_vectorize_in_table_from_dataset(self):
        pxt.drop_dir("unit_test", force=True)

        pxt.create_dir("unit_test", if_exists="ignore")
        table_path = "unit_test.vectorize_from_dataset"

        dataset = DummyDataset(dataset_len=2)

        gv = GoldVectorizer(
            table_path=table_path,
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="features",
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

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_table_from_table(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        source_rows = [
            {"idx": 0, "features": torch.zeros(4, 3).numpy(), "label": "dummy"},
            {"idx": 1, "features": torch.zeros(4, 3).numpy(), "label": "dummy"},
        ]

        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="features",
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

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_table_without_idx(
        self,
    ):
        pxt.drop_dir("unit_test", force=True)

        def collate_fn(batch):
            data = torch.stack([b["features"] for b in batch], dim=0)
            return {"features": data}

        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=TensorVectorizer(),
            collate_fn=collate_fn,
            data_key="features",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )

        table = gv.vectorize_in_table(
            DummyDataset(dataset_len=2),
        )
        assert table.count() == 128
        for i, row in enumerate(table.collect()):
            assert row["idx"] == i
            if i < 64:
                assert row["idx_sample"] == 0
            else:
                assert row["idx_sample"] == 1
            assert row["vectorized"].shape == (3,)

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_table_with_non_dict_item(
        self,
    ):
        pxt.drop_dir("unit_test", force=True)

        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=TensorVectorizer(),
            collate_fn=lambda x: [d["features"] for d in x],
            data_key="features",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample must be a dictionary"):
            gv.vectorize_in_table(DummyDataset())

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_table_with_missing_data_key(
        self,
    ):
        pxt.drop_dir("unit_test", force=True)

        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="not_present",
            vectorized_key="vectorized",
            batch_size=1,
            num_workers=0,
            allow_existing=False,
        )
        with pytest.raises(ValueError, match="Sample is missing expected keys"):
            gv.vectorize_in_table(DummyDataset())

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_table_with_max_batches(
        self,
    ):
        pxt.drop_dir("unit_test", force=True)

        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="features",
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
            assert row["idx"] == i

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_dataset(self):
        pxt.drop_dir("unit_test", force=True)

        pxt.create_dir("unit_test", if_exists="ignore")
        table_path = "unit_test.vectorize_dataset"

        dataset = DummyDataset(dataset_len=2)

        gv = GoldVectorizer(
            table_path=table_path,
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="features",
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

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_dataset_from_table(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        source_rows = [
            {"idx": 0, "features": torch.zeros(3, 8, 8).numpy()},
            {"idx": 1, "features": torch.zeros(3, 8, 8).numpy()},
        ]
        pxt.create_dir("unit_test", if_exists="ignore")
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        gv = GoldVectorizer(
            table_path=desc_path,
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="features",
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

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_table_after_restart(
        self,
    ):
        pxt.drop_dir("unit_test", force=True)

        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="features",
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
            assert row["idx"] == i

        gv.max_batches = None
        vectorized_table = gv.vectorize_in_table(dataset)

        assert vectorized_table.count() == 640
        for i, row in enumerate(vectorized_table.collect()):
            assert row["idx"] == i
            assert row["vectorized"].shape == (3,)

        pxt.drop_dir("unit_test", force=True)

    def test_vectorize_in_table_after_restart_with_restart_disallowed(
        self,
    ):
        pxt.drop_dir("unit_test", force=True)

        gv = GoldVectorizer(
            table_path="unit_test.vectorize",
            vectorizer=TensorVectorizer(),
            collate_fn=None,
            data_key="features",
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

        pxt.drop_dir("unit_test", force=True)
