import torch
import pytest
from goldener.utils import check_x_and_y_shapes


class TestCheckXAndYShapes:
    def test_1d_shapes_match(self):
        x = torch.zeros(5)
        y = torch.zeros(5)
        check_x_and_y_shapes(x.shape, y.shape)  # Should not raise

    def test_1d_shapes_mismatch(self):
        x = torch.zeros(5)
        y = torch.zeros(6)
        with pytest.raises(ValueError):
            check_x_and_y_shapes(x.shape, y.shape)

    def test_2d_shapes_match_channel1(self):
        x = torch.zeros(3, 4)
        y = torch.zeros(3, 1)
        check_x_and_y_shapes(x.shape, y.shape)  # Should not raise

    def test_2d_shapes_mismatch_channel(self):
        x = torch.zeros(3, 4)
        y = torch.zeros(3, 2)
        with pytest.raises(ValueError):
            check_x_and_y_shapes(x.shape, y.shape)

    def test_2d_shapes_mismatch_batch(self):
        x = torch.zeros(3, 4)
        y = torch.zeros(4, 1)
        with pytest.raises(ValueError):
            check_x_and_y_shapes(x.shape, y.shape)

    def test_2d_shapes_match_batch_y_one(self):
        x = torch.zeros(3, 4)
        y = torch.zeros(1, 1)
        check_x_and_y_shapes(x.shape, y.shape)  # Should not raise

    def test_3d_shapes_match(self):
        x = torch.zeros(2, 3, 4)
        y = torch.zeros(2, 1, 4)
        check_x_and_y_shapes(x.shape, y.shape)  # Should not raise

    def test_3d_shapes_mismatch_channel(self):
        x = torch.zeros(2, 3, 4)
        y = torch.zeros(2, 2, 4)
        with pytest.raises(ValueError):
            check_x_and_y_shapes(x.shape, y.shape)

    def test_3d_shapes_mismatch_batch(self):
        x = torch.zeros(2, 3, 4)
        y = torch.zeros(3, 1, 4)
        with pytest.raises(ValueError):
            check_x_and_y_shapes(x.shape, y.shape)

    def test_3d_shapes_mismatch_last_dims(self):
        x = torch.zeros(2, 3, 4)
        y = torch.zeros(2, 1, 5)
        with pytest.raises(ValueError):
            check_x_and_y_shapes(x.shape, y.shape)

    def test_3d_shapes_match_batch_y_one(self):
        x = torch.zeros(2, 3, 4)
        y = torch.zeros(1, 1, 4)
        check_x_and_y_shapes(x.shape, y.shape)  # Should not raise

    def test_y_is_none(self):
        x = torch.zeros(2, 3, 4)
        y = None
        # Should not raise if y is None (skip check)
        if y is not None:
            check_x_and_y_shapes(x.shape, y.shape)
