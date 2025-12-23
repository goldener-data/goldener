import torch
import pytest
from goldener.utils import (
    check_x_and_y_shapes,
    get_size_and_sampling_count_per_chunk,
    check_sampling_size,
    check_all_same_type,
    get_sampling_count_from_size,
)


class TestCheckXAndYShapes:
    def test_1d_shapes_match(self):
        x = torch.zeros(5)
        y = torch.zeros(5)
        check_x_and_y_shapes(x.shape, y.shape)

    def test_1d_shapes_mismatch(self):
        x = torch.zeros(5)
        y = torch.zeros(6)
        with pytest.raises(ValueError):
            check_x_and_y_shapes(x.shape, y.shape)

    def test_2d_shapes_match_channel1(self):
        x = torch.zeros(3, 4)
        y = torch.zeros(3, 1)
        check_x_and_y_shapes(x.shape, y.shape)

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
        check_x_and_y_shapes(x.shape, y.shape)

    def test_3d_shapes_match(self):
        x = torch.zeros(2, 3, 4)
        y = torch.zeros(2, 1, 4)
        check_x_and_y_shapes(x.shape, y.shape)

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
        check_x_and_y_shapes(x.shape, y.shape)

    def test_y_is_none(self):
        x = torch.zeros(2, 3, 4)
        y = None

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


class TestCheckSamplingSizes:
    def test_valid_integer_sampling(self):
        check_size = 5
        total_size = 10
        check_sampling_size(check_size, total_size)

    def test_invalid_integer_sampling_raises(self):
        check_size = 15
        total_size = 10
        with pytest.raises(
            ValueError,
            match="Sampling size as int must be greater than 0 and less than the total number of samples",
        ):
            check_sampling_size(check_size, total_size)

    def test_valid_float_sampling(self):
        check_size = 0.5
        check_sampling_size(check_size)

    def test_invalid_float_sampling_raises(self):
        check_size = 1.5
        with pytest.raises(
            ValueError,
            match="Sampling size as float must be greater than 0.0 and at most 1.0",
        ):
            check_sampling_size(check_size)


class TestCheckAllSameType:
    def test_all_same_type(self):
        check_all_same_type([1, 2, 3])

    def test_not_all_same_type(self):
        items = [1, "2", 3]
        with pytest.raises(TypeError, match="All elements must be of the same type"):
            check_all_same_type(items)


class TestGetSamplingCountFromSize:
    def test_integer_sampling_size(self):
        sampling_size = 5
        total_size = 10
        count = get_sampling_count_from_size(
            sampling_size,
            total_size,
        )
        assert count == 5

    def test_float_sampling_size(self):
        sampling_size = 0.5
        total_size = 10
        count = get_sampling_count_from_size(
            sampling_size,
            total_size,
        )
        assert count == 5

    def test_invalid_integer_sampling_raises(self):
        sampling_size = 0
        total_size = 10
        with pytest.raises(
            ValueError, match="Sampling size as int must be greater than 0"
        ):
            get_sampling_count_from_size(
                sampling_size,
                total_size,
            )

        with pytest.raises(
            ValueError,
            match="Sampling size as int must be less than the total number of ",
        ):
            get_sampling_count_from_size(
                10,
                total_size,
            )

    def test_invalid_float_sampling_raises(self):
        sampling_size = 1.0001
        total_size = 10
        with pytest.raises(
            ValueError,
            match="Sampling size as float must be greater than 0.0 and at most 1.0",
        ):
            get_sampling_count_from_size(
                sampling_size,
                total_size,
            )

        with pytest.raises(
            ValueError,
            match="Total size must be provided when sampling size is a float",
        ):
            get_sampling_count_from_size(
                0.5,
                None,
            )
