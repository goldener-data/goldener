import torch
import pytest
import pixeltable as pxt
from goldener.vectorize import (
    TensorVectorizer,
    Filter2DWithCount,
    FilterLocation,
    GoldVectorizer,
    Vectorized,
    unwrap_vectors_in_batch,
    vectorize_and_unwrap_in_batch,
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

    def test_vectorize_with_y_and_full_zero(self):
        x = self.make_tensor((2, 2, 5))
        y = torch.zeros(2, 2)
        v = TensorVectorizer(channel_pos=2)
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (4, 5)

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

    def test_vectorize_with_random(self):
        x = self.make_tensor()
        v = TensorVectorizer(
            random=Filter2DWithCount(
                filter_count=1, filter_location=FilterLocation.RANDOM, keep=True
            )
        )
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

    def test_vectorizer_invalid_keep_type_random(self):
        # keep filter cannot be random
        keep = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.RANDOM,
            keep=True,
        )
        with pytest.raises(ValueError, match="keep"):
            TensorVectorizer(keep=keep)

    def test_vectorizer_invalid_keep_type_not_keeping(self):
        # keep filter must have keep=True
        keep = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.START,
            keep=False,
        )
        with pytest.raises(ValueError, match="keep"):
            TensorVectorizer(keep=keep)

    def test_vectorizer_invalid_remove_type_random(self):
        # remove filter cannot be random
        remove = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.RANDOM,
            keep=False,
        )
        with pytest.raises(ValueError, match="remove"):
            TensorVectorizer(remove=remove)

    def test_vectorizer_invalid_remove_type_not_removing(self):
        # remove filter must have keep=False
        remove = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.START,
            keep=True,
        )
        with pytest.raises(ValueError, match="remove"):
            TensorVectorizer(remove=remove)

    def test_vectorizer_invalid_random_type_not_random(self):
        # random filter must be random
        rand = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.START,
            keep=True,
        )
        with pytest.raises(ValueError, match="random"):
            TensorVectorizer(random=rand)

    def test_vectorizer_invalid_random_type_not_keeping(self):
        # random filter must have keep=True so it selects indices to keep
        rand = Filter2DWithCount(
            filter_count=1,
            filter_location=FilterLocation.RANDOM,
            keep=False,
        )
        with pytest.raises(ValueError, match="random"):
            TensorVectorizer(random=rand)

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
        v = TensorVectorizer(keep=keep, remove=remove, random=rand)
        x = self.make_tensor()
        _ = v.vectorize(x)


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

    def test_vectorize_in_table_with_target(self):
        pxt.drop_dir("unit_test", force=True)

        src_path = "unit_test.src_table_vectorize"
        desc_path = "unit_test.vectorize_from_table"

        source_rows = [
            {
                "idx": 0,
                "features": torch.zeros(4, 3).numpy(),
                "label": "dummy",
                "target": torch.ones(1, 3).numpy(),
            },
            {
                "idx": 1,
                "features": torch.zeros(4, 3).numpy(),
                "label": "dummy",
                "target": torch.ones(1, 3).numpy(),
            },
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
        idx_vector = set()
        idx = set()
        for row in table.collect():
            idx.add(row["idx"])
            idx_vector.add(row["idx_vector"])

        assert idx == {0, 1}
        assert idx_vector == set(range(128))

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
            assert row["idx_vector"] == i

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
            assert row["idx_vector"] == i

        gv.max_batches = None
        vectorized_table = gv.vectorize_in_table(dataset)

        assert vectorized_table.count() == 640
        for i, row in enumerate(vectorized_table.collect()):
            assert row["idx_vector"] == i
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


class TestUnwrapVectorsInBatch:
    def make_vectorized(self, n_vectors: int = 4, dim: int = 5):
        vectors = torch.arange(n_vectors * dim, dtype=torch.float32).reshape(
            n_vectors, dim
        )
        batch_indices = torch.tensor([0, 0, 1, 1])[:n_vectors]
        return Vectorized(vectors=vectors, batch_indices=batch_indices)

    def make_batch(self, batch_size: int = 2):
        return {"idx": list(range(batch_size)), "label": ["a", "b"][:batch_size]}

    def test_basic_output_keys(self):
        vectorized = self.make_vectorized()
        batch = self.make_batch()
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch)
        assert set(result.keys()) == {"idx", "vectorized", "idx_vector"}

    def test_output_length_matches_vectors(self):
        vectorized = self.make_vectorized()
        batch = self.make_batch()
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch)
        assert len(result["vectorized"]) == 4
        assert len(result["idx"]) == 4
        assert len(result["idx_vector"]) == 4

    def test_idx_vector_values_with_default_starts(self):
        vectorized = self.make_vectorized()
        batch = self.make_batch()
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch)
        assert result["idx_vector"] == [0, 1, 2, 3]

    def test_idx_vector_values_with_nonzero_starts(self):
        vectorized = self.make_vectorized()
        batch = self.make_batch()
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch, starts=10)
        assert result["idx_vector"] == [10, 11, 12, 13]

    def test_idx_maps_to_original_samples(self):
        vectorized = self.make_vectorized()
        batch = {"idx": [42, 99]}
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch)
        # batch_indices = [0, 0, 1, 1] → original idx [42, 42, 99, 99]
        assert result["idx"] == [42, 42, 99, 99]

    def test_to_keep_adds_keys(self):
        vectorized = self.make_vectorized()
        batch = self.make_batch()
        result = unwrap_vectors_in_batch(
            vectorized, "vectorized", batch, to_keep=["label"]
        )
        assert "label" in result
        assert result["label"] == ["a", "a", "b", "b"]

    def test_custom_vectorized_key(self):
        vectorized = self.make_vectorized()
        batch = self.make_batch()
        result = unwrap_vectors_in_batch(vectorized, "embeddings", batch)
        assert "embeddings" in result
        assert "vectorized" not in result

    def test_vectors_are_preserved(self):
        vectorized = self.make_vectorized()
        batch = self.make_batch()
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch)
        for i, vec in enumerate(result["vectorized"]):
            assert torch.equal(vec, vectorized.vectors[i])

    def test_idx_tensor_values(self):
        # idx values are tensors (as returned by a DataLoader)
        vectorized = self.make_vectorized()
        batch = {"idx": [torch.tensor(5), torch.tensor(7)]}
        result = unwrap_vectors_in_batch(vectorized, "vectorized", batch)
        assert result["idx"] == [5, 5, 7, 7]


class TestVectorizeAndUnwrapInBatch:
    def make_batch(self, batch_size: int = 2, channels: int = 3, rows: int = 4):
        data = torch.zeros(batch_size, rows, channels)
        return {"idx": list(range(batch_size)), "data": data}

    def test_basic_output_keys(self):
        batch = self.make_batch()
        vectorizer = TensorVectorizer(channel_pos=2)
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key=None,
        )
        assert set(result.keys()) == {"idx", "vectorized", "idx_vector"}

    def test_output_length(self):
        batch = self.make_batch(batch_size=2, channels=3, rows=4)
        vectorizer = TensorVectorizer(channel_pos=2)
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key=None,
        )
        # 2 samples × 4 rows each
        assert len(result["vectorized"]) == 8

    def test_with_starts(self):
        batch = self.make_batch()
        vectorizer = TensorVectorizer(channel_pos=2)
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key=None,
            starts=100,
        )
        assert result["idx_vector"][0] == 100

    def test_with_to_keep(self):
        batch = self.make_batch()
        batch["label"] = ["x", "y"]
        vectorizer = TensorVectorizer(channel_pos=2)
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key=None,
            to_keep=["label"],
        )
        assert "label" in result
        assert len(result["label"]) == len(result["vectorized"])

    def test_with_target_key_present(self):
        batch_size, rows, channels = 2, 4, 3
        data = torch.ones(batch_size, rows, channels)
        # target: keep only row 0 for each sample
        target = torch.zeros(batch_size, 1, channels)
        batch = {"idx": list(range(batch_size)), "data": data, "target": target}
        vectorizer = TensorVectorizer(channel_pos=2)
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key="target",
        )
        # target is all zeros so all rows are filtered out; fallback keeps all
        assert len(result["vectorized"]) > 0

    def test_with_missing_target_key(self):
        batch = self.make_batch()
        vectorizer = TensorVectorizer(channel_pos=2)
        # target_key not in batch → should behave as if target is None
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="vectorized",
            target_key="missing_key",
        )
        assert len(result["vectorized"]) == 8

    def test_custom_vectorized_key(self):
        batch = self.make_batch()
        vectorizer = TensorVectorizer(channel_pos=2)
        result = vectorize_and_unwrap_in_batch(
            batch=batch,
            vectorizer=vectorizer,
            data_key="data",
            vectorized_key="embeddings",
            target_key=None,
        )
        assert "embeddings" in result
        assert "vectorized" not in result
