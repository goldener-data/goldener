import torch
import pytest
from goldener.utils import (
    check_x_and_y_shapes,
    get_size_and_sampling_count_per_chunk,
    check_sampling_size,
    check_all_same_type,
    get_ratio_list_sum,
    get_ratios_for_counts,
    filter_batch_from_indices,
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


class TestGetRatioListSum:
    def test_valid_ratios_sum_to_one(self):
        ratios = [0.3, 0.3, 0.4]
        result = get_ratio_list_sum(ratios)
        assert result == 1.0

    def test_valid_ratios_sum_less_than_one(self):
        ratios = [0.2, 0.3, 0.4]
        result = get_ratio_list_sum(ratios)
        assert result == 0.9

    def test_single_ratio_one(self):
        ratios = [1.0]
        result = get_ratio_list_sum(ratios)
        assert result == 1.0

    def test_single_ratio_less_than_one(self):
        ratios = [0.5]
        result = get_ratio_list_sum(ratios)
        assert result == 0.5

    def test_invalid_ratios_sum_zero(self):
        ratios = [0.0]
        with pytest.raises(ValueError):
            get_ratio_list_sum(ratios)

    def test_invalid_ratios_sum_negative(self):
        ratios = [-0.1, 0.5]
        with pytest.raises(ValueError):
            get_ratio_list_sum(ratios)

    def test_invalid_ratios_sum_greater_than_one(self):
        ratios = [0.6, 0.6]
        with pytest.raises(ValueError):
            get_ratio_list_sum(ratios)

    def test_empty_list(self):
        ratios = []
        with pytest.raises(ValueError):
            get_ratio_list_sum(ratios)


class TestGetRatiosForCounts:
    def test_simple_counts(self):
        counts = [10, 20, 30]
        ratios = get_ratios_for_counts(counts)
        assert ratios == [10 / 60, 20 / 60, 30 / 60]

    def test_equal_counts(self):
        counts = [5, 5, 5, 5]
        ratios = get_ratios_for_counts(counts)
        assert ratios == [0.25, 0.25, 0.25, 0.25]

    def test_single_count(self):
        counts = [100]
        ratios = get_ratios_for_counts(counts)
        assert ratios == [1.0]

    def test_counts_with_zeros(self):
        counts = [0, 10, 20]
        ratios = get_ratios_for_counts(counts)
        assert ratios == [0.0, 10 / 30, 20 / 30]

    def test_large_counts(self):
        counts = [1000, 2000, 3000]
        ratios = get_ratios_for_counts(counts)
        assert ratios == [1000 / 6000, 2000 / 6000, 3000 / 6000]

    def test_ratios_sum_to_one(self):
        counts = [7, 13, 20]
        ratios = get_ratios_for_counts(counts)
        assert abs(sum(ratios) - 1.0) < 1e-9  # Account for floating point precision


class TestFilterBatchFromIndices:
    def test_filter_some_indices(self):
        batch = {
            "idx": [0, 1, 2, 3, 4],
            "data": ["a", "b", "c", "d", "e"],
        }
        to_remove = {1, 3}
        result = filter_batch_from_indices(batch, to_remove)
        assert result["idx"] == [0, 2, 4]
        assert result["data"] == ["a", "c", "e"]

    def test_filter_no_indices(self):
        batch = {
            "idx": [0, 1, 2],
            "data": ["a", "b", "c"],
        }
        to_remove = set()
        result = filter_batch_from_indices(batch, to_remove)
        assert result["idx"] == [0, 1, 2]
        assert result["data"] == ["a", "b", "c"]

    def test_filter_all_indices(self):
        batch = {
            "idx": [0, 1, 2],
            "data": ["a", "b", "c"],
        }
        to_remove = {0, 1, 2}
        result = filter_batch_from_indices(batch, to_remove)
        assert result == {}

    def test_filter_with_torch_tensors(self):
        batch = {
            "idx": torch.tensor([0, 1, 2, 3]),
            "features": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        }
        to_remove = {1, 3}
        result = filter_batch_from_indices(batch, to_remove)
        assert torch.equal(result["idx"], torch.tensor([0, 2]))
        assert torch.equal(result["features"], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

    def test_filter_with_mixed_types(self):
        batch = {
            "idx": torch.tensor([0, 1, 2]),
            "data": ["a", "b", "c"],
            "values": [10, 20, 30],
        }
        to_remove = {1}
        result = filter_batch_from_indices(batch, to_remove)
        assert torch.equal(result["idx"], torch.tensor([0, 2]))
        assert result["data"] == ["a", "c"]
        assert result["values"] == [10, 30]

    def test_filter_with_custom_index_key(self):
        batch = {
            "my_idx": [0, 1, 2, 3],
            "data": ["a", "b", "c", "d"],
        }
        to_remove = {0, 2}
        result = filter_batch_from_indices(batch, to_remove, index_key="my_idx")
        assert result["my_idx"] == [1, 3]
        assert result["data"] == ["b", "d"]

    def test_filter_with_torch_tensor_indices(self):
        batch = {
            "idx": torch.tensor([10, 20, 30, 40]),
            "data": ["a", "b", "c", "d"],
        }
        to_remove = {20, 40}
        result = filter_batch_from_indices(batch, to_remove)
        assert torch.equal(result["idx"], torch.tensor([10, 30]))
        assert result["data"] == ["a", "c"]

    def test_filter_empty_batch(self):
        batch = {
            "idx": [],
            "data": [],
        }
        to_remove = {0, 1}
        result = filter_batch_from_indices(batch, to_remove)
        assert result == {}


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
