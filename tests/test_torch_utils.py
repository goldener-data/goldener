import numpy as np
import torch
import pytest
from collections import Counter
from goldener.torch_utils import (
    torch_tensor_to_numpy_vectors,
    numpy_vectors_to_torch_tensor,
    np_transform_from_torch,
    make_2d_tensor,
    ResetableTorchIterableDataset,
    get_unique_values_in_tensor,
    shuffle_list,
    get_subset_indices_for_indices,
)


class TestTorchTensorToNumpyVectors:
    def test_torch_tensor_to_numpy_vectors_0d(self):
        t = torch.tensor(3)
        arr = torch_tensor_to_numpy_vectors(t)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 1)

    def test_torch_tensor_to_numpy_vectors_1d(self):
        t = torch.rand(3)
        arr = torch_tensor_to_numpy_vectors(t)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 1)

    def test_torch_tensor_to_numpy_vectors_2d(self):
        t = torch.rand(3, 2)
        arr = torch_tensor_to_numpy_vectors(t)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)

    def test_torch_tensor_to_numpy_vectors_3d(self):
        t = torch.rand(2, 3, 4, dtype=torch.float32)
        arr = torch_tensor_to_numpy_vectors(t)
        assert arr.shape == (
            8,
            3,
        )


class TestNumpyVectorsToTorchTensor:
    def test_numpy_vectors_to_torch_tensor_0d(self):
        arr = np.array(3)
        shape = (1, 1)
        t = numpy_vectors_to_torch_tensor(
            arr, shape, torch.float32, torch.device("cpu")
        )
        assert isinstance(t, torch.Tensor)
        assert t.shape == shape
        assert t.device.type == "cpu"
        assert t.dtype == torch.float32

    def test_numpy_vectors_to_torch_tensor_1d(self):
        arr = np.random.randint(0, 10, size=(3,), dtype=np.int32)
        shape = (3, 1)
        t = numpy_vectors_to_torch_tensor(
            arr, shape, torch.float32, torch.device("cpu")
        )
        assert isinstance(t, torch.Tensor)
        assert t.shape == shape
        assert t.device.type == "cpu"
        assert t.dtype == torch.float32

    def test_numpy_vectors_to_torch_tensor_2d(self):
        arr = np.random.randint(0, 10, size=(3, 2), dtype=np.int32)
        shape = (3, 2)
        t = numpy_vectors_to_torch_tensor(
            arr, shape, torch.float32, torch.device("cpu")
        )
        assert isinstance(t, torch.Tensor)
        assert t.shape == shape
        assert t.device.type == "cpu"
        assert t.dtype == torch.float32

    def test_numpy_vectors_to_torch_tensor_3d(self):
        arr = np.random.rand(2, 4, 3).astype(np.float32)
        shape = (2, 3, 4)
        t = numpy_vectors_to_torch_tensor(
            arr, shape, torch.float32, torch.device("cpu")
        )
        assert isinstance(t, torch.Tensor)
        assert t.shape == shape
        assert t.device.type == "cpu"
        assert t.dtype == torch.float32

    def test_numpy_vectors_to_torch_tensor_invalid_shape(self):
        arr = np.random.rand(
            3,
        )
        with pytest.raises(ValueError):
            numpy_vectors_to_torch_tensor(arr, (3,), torch.float32, torch.device("cpu"))


class TestNpTransformFromTorch:
    def test_np_transform_from_torch(self):
        def dummy_transform(x: np.ndarray) -> np.ndarray:
            return x + 1

        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = np_transform_from_torch(t, dummy_transform)
        assert torch.allclose(t + 1, out)


class TestMake2DTensor:
    def test_0d_tensor(self):
        t = torch.tensor(7)
        out = make_2d_tensor(t)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 1)
        assert out[0, 0] == 7

    def test_1d_tensor(self):
        t = torch.tensor([1, 2, 3])
        out = make_2d_tensor(t)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 1)
        assert torch.equal(out.squeeze(1), t)

    def test_2d_tensor(self):
        t = torch.arange(6).reshape(2, 3)
        out = make_2d_tensor(t)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3)
        assert torch.equal(out, t)

    def test_3d_tensor(self):
        t = torch.arange(24).reshape(2, 3, 4)
        out = make_2d_tensor(t)
        # After moveaxis(1, -1): shape (2, 4, 3), then flatten(0, -2): (8, 3)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (8, 3)
        # Check content: first block should match t[0].moveaxis(0, -1).reshape(-1, 3)
        t_moved = t.moveaxis(1, -1)
        t_flat = t_moved.flatten(0, -2)
        assert torch.equal(out, t_flat)

    def test_4d_tensor(self):
        t = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        out = make_2d_tensor(t)
        # After moveaxis(1, -1): shape (2, 4, 5, 3), then flatten(0, -2): (40, 3)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (40, 3)
        t_moved = t.moveaxis(1, -1)
        t_flat = t_moved.flatten(0, -2)
        assert torch.equal(out, t_flat)


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data: list[int]):
        super().__init__()
        self.data = data

    def __iter__(self):
        return iter(self.data)


class TestResetableTorchIterableDataset:
    def test_simple_usage(self):
        dataset = ResetableTorchIterableDataset(DummyIterableDataset(list(range(10))))
        out = list(dataset)
        assert out == list(range(10))

    def test_reset(self):
        dataset = ResetableTorchIterableDataset(DummyIterableDataset(list(range(10))))
        start = next(iter(dataset))
        dataset.reset()
        start_after_reset = next(iter(dataset))
        assert start_after_reset == start


class TestGetUniqueValuesInTensor:
    def test_unique_along_dim1_vectors(self):
        t = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
        t[0, :, 0, 0] = torch.tensor([1, 2, 3])
        t[1, :, 3, 3] = torch.tensor([3, 2, 1])
        out = get_unique_values_in_tensor(t, dim=1)
        expected = torch.tensor([[0, 0, 0], [1, 2, 3], [3, 2, 1]], dtype=torch.uint8)
        assert (out == expected).all()

    def test_unique_along_last_dim_scalars(self):
        t = torch.zeros(
            (
                2,
                3,
            ),
            dtype=torch.uint8,
        )
        t[
            0,
            :,
        ] = torch.tensor([1, 2, 3])
        out = get_unique_values_in_tensor(t, dim=-1)
        expected = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.uint8)
        assert (out == expected).all()

    def test_dim0_behavior(self):
        t = torch.zeros((2,), dtype=torch.uint8)
        t[0] = torch.tensor(3)
        out = get_unique_values_in_tensor(t, dim=0)
        expected = torch.tensor([3, 0], dtype=torch.uint8)
        assert (out == expected).all()


class TestShuffleList:
    def test_same_seed_same_shuffle(self):
        items = list(range(20))
        gen_a = torch.Generator().manual_seed(123)
        gen_b = torch.Generator().manual_seed(123)

        out_a = shuffle_list(items, generator=gen_a)
        out_b = shuffle_list(items, generator=gen_b)

        assert out_a == out_b

    def test_same_elements(self):
        items = list(range(20))
        out = shuffle_list(items, generator=torch.Generator().manual_seed(123))
        assert Counter(items) == Counter(out)

    def test_does_not_mutate_input_list(self):
        items = [10, 11, 12, 13]
        items_before = list(items)

        shuffle_list(items, generator=torch.Generator().manual_seed(1))

        assert items == items_before

    def test_empty_list(self):
        out = shuffle_list([], generator=torch.Generator().manual_seed(0))
        assert out == []

    def test_singleton_list(self):
        out = shuffle_list([42], generator=torch.Generator().manual_seed(0))
        assert out == [42]


class TestGetSubsetIndicesForIndices:
    def test_basic(self):
        result = get_subset_indices_for_indices(
            indices_in_dataset={10, 20},
            indices_in_subset=[10, 20, 30],
        )
        assert sorted(result) == [0, 1]

    def test_multiple_duplicates_in_subset(self):
        result = get_subset_indices_for_indices(
            indices_in_dataset={5, 7},
            indices_in_subset=[5, 7, 7, 5],
        )
        assert sorted(result) == [0, 1, 2, 3]

    def test_raises_when_dataset_index_not_in_subset(self):
        with pytest.raises(
            ValueError,
            match="The indices from the dataset are required to be all present in the subset",
        ):
            get_subset_indices_for_indices(
                indices_in_dataset={99},
                indices_in_subset=[0, 1, 2],
            )

    def test_empty_dataset_indices(self):
        result = get_subset_indices_for_indices(
            indices_in_dataset=set(),
            indices_in_subset=[0, 1, 2],
        )
        assert result == []
