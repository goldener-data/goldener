import torch
import pytest
from goldener.utils import check_x_and_y_shapes, get_size_and_sampling_count_per_chunk


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


class TestGetSizeAndSamplingCountPerChunk:
    def test_single_chunk_when_max_ge_total(self):
        total_size = 100
        sampling_size = 10
        max_chunk_size = 100
        sizes, samplings = get_size_and_sampling_count_per_chunk(
            total_size, sampling_size, max_chunk_size
        )
        assert sizes == [100]
        assert samplings == [10]

    def test_multiple_chunks_even_split(self):
        total_size = 100
        sampling_size = 10
        max_chunk_size = 30
        sizes, samplings = get_size_and_sampling_count_per_chunk(
            total_size, sampling_size, max_chunk_size
        )
        assert sizes == [25, 25, 25, 25]
        assert samplings == [2, 2, 2, 4]

    def test_sampling_greater_than_total_raises(self):
        with pytest.raises(ValueError):
            get_size_and_sampling_count_per_chunk(10, 11, 5)

    def test_edge_case_small_numbers(self):
        total_size = 5
        sampling_size = 2
        max_chunk_size = 2
        sizes, samplings = get_size_and_sampling_count_per_chunk(
            total_size, sampling_size, max_chunk_size
        )
        assert sizes == [1, 1, 3]
        assert samplings == [0, 0, 2]
