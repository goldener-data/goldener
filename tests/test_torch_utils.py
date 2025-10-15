import numpy as np
import torch
import pytest
from goldener.torch_utils import (
    torch_tensor_to_numpy_vectors,
    numpy_vectors_to_torch_tensor,
    np_transform_from_torch,
    make_2d_tensor,
)


class TestTensorToVector:
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


class TestArrayToTensor:
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
        shape = (2, 3, 4)  # Desired shape with channel moved to 1st dimension
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


class TestNpTranformFromTorch:
    def test_np_transform_from_torch(self):
        def dummy_transform(x: np.ndarray) -> np.ndarray:
            return x + 1

        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = np_transform_from_torch(t, dummy_transform)
        assert out.shape == t.shape


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
