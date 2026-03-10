import torch
import pytest

from goldener.distance import cosine_distance


class TestCosineDistance:
    def test_basic_behavior(self) -> None:
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        d = cosine_distance(x, y)

        assert d.shape == (2, 2)

        assert torch.allclose(d[0, 0], torch.tensor(0.0))
        assert torch.allclose(d[1, 1], torch.tensor(0.0))

        assert torch.allclose(d[0, 1], torch.tensor(1.0))
        assert torch.allclose(d[1, 0], torch.tensor(1.0))

    def test_broadcasting_shapes(self) -> None:
        x = torch.randn(5, 8)
        y = torch.randn(3, 8)

        d = cosine_distance(x, y)

        assert d.shape == (5, 3)

    def test_zero_vectors_handling(self) -> None:
        x = torch.zeros(2, 4)
        y = torch.zeros(3, 4)

        d = cosine_distance(x, y)

        assert d.shape == (2, 3)
        assert torch.isfinite(d).all()

    def test_inputs_must_be_2d(self) -> None:
        x_1d = torch.tensor([1.0, 0.0])
        x_2d = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Input tensors must be 2D"):
            cosine_distance(x_1d, x_2d)

        with pytest.raises(ValueError, match="Input tensors must be 2D"):
            cosine_distance(x_2d, x_1d)

    def test_inputs_must_same_dim(self) -> None:
        x1 = torch.rand(2, 4)
        x2 = torch.rand(3, 5)

        with pytest.raises(ValueError, match="must have the same number of channels"):
            cosine_distance(x1, x2)
