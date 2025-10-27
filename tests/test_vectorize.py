import torch
import pytest
from goldener.vectorize import GoldVectorizer, Filter2DWithCount, FilterLocation


class TestGoldVectorizer:
    def make_tensor(self, shape=(2, 5, 2)):
        return torch.randint(0, 100, shape)

    def test_vectorize_no_y(self):
        x = self.make_tensor()
        v = GoldVectorizer()
        vec = v.vectorize(x)
        assert vec.vectors.shape == (4, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 0, 1, 1]))

    def test_vectorize_with_y(self):
        x = self.make_tensor()
        y = torch.ones(2, 1, 2)
        y[0, 0, 0] = 0
        v = GoldVectorizer()
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (3, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1, 1]))

    def test_vectorize_with_keep(self):
        x = self.make_tensor()
        keep = Filter2DWithCount(
            filter_count=1, filter_location=FilterLocation.START, keep=True
        )
        v = GoldVectorizer(keep=keep)
        vec = v.vectorize(x)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_with_remove(self):
        x = self.make_tensor()
        remove = Filter2DWithCount(
            filter_count=1, filter_location=FilterLocation.END, keep=False
        )
        v = GoldVectorizer(remove=remove)
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
        v = GoldVectorizer(keep=keep, remove=remove)
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

        v = GoldVectorizer(transform_y=transform_y)
        vec = v.vectorize(x, y)
        assert vec.vectors.shape == (2, 5)
        assert torch.equal(vec.batch_indices, torch.tensor([0, 1]))

    def test_vectorize_shape_mismatch(self):
        x = self.make_tensor()
        y = torch.ones(2, 1, 3)
        v = GoldVectorizer()
        with pytest.raises(ValueError):
            v.vectorize(x, y)

    def test_vectorize_2d_input(self):
        x = self.make_tensor((4, 5))
        v = GoldVectorizer()
        with pytest.raises(ValueError):
            v.vectorize(x)

    def test_vectorizer_invalid_keep_type(self):
        with pytest.raises(ValueError):
            GoldVectorizer(keep=Filter2DWithCount())

    def test_vectorizer_invalid_remove_type(self):
        with pytest.raises(ValueError):
            GoldVectorizer(remove=Filter2DWithCount())

    def test_vectorizer_invalid_random_type(self):
        with pytest.raises(ValueError):
            GoldVectorizer(
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
