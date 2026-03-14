import torch
import pytest
from goldener.utils import (
    check_x_and_y_shapes,
    get_size_and_sampling_count_per_chunk,
    get_ratio_list_sum,
    get_ratios_for_counts,
    filter_batch_from_indices,
    get_indices_with_labels,
    check_sampling_size,
    check_all_same_type,
    get_sampling_count_from_size,
    split_sampling_among_chunks,
    transform_batch_from_multiple_to_binarized_targets,
    get_sampling_count_from_ratios,
    transform_batch_from_multilabel_to_independent_labels,
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


class TestCheckSamplingSize:
    def test_valid_integer_sampling(self):
        total_size = 10
        check_sampling_size(5, total_size)
        check_sampling_size(10, total_size, force_max=True)

    def test_invalid_integer_sampling_raises(self):
        with pytest.raises(
            ValueError,
            match="Sampling size as int must be greater than 0 and less or equal than the total number of samples",
        ):
            check_sampling_size(15, 10)

        with pytest.raises(
            ValueError,
            match="Sampling size as int must be equal to the total number of samples",
        ):
            check_sampling_size(14, 15, force_max=True)

    def test_valid_float_sampling(self):
        check_sampling_size(0.5)
        check_sampling_size(1.0, force_max=True)

    def test_invalid_float_sampling_raises(self):
        with pytest.raises(
            ValueError,
            match="Sampling size as float must be greater than 0.0 and at most 1.0",
        ):
            check_sampling_size(1.5)

        with pytest.raises(
            ValueError,
            match="Sampling size as float must be equal to 1.0",
        ):
            check_sampling_size(0.99, force_max=True)


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


class TestGetRatioListSum:
    def test_valid_ratios_sum_to_one(self):
        ratios = [0.3, 0.3, 0.4]
        result = get_ratio_list_sum(ratios)
        assert result == 1.0

    def test_valid_ratios_sum_less_than_one(self):
        ratios = [0.2, 0.3, 0.4]
        result = get_ratio_list_sum(ratios)
        assert result == 0.9

    def test_invalid_ratios_sum_zero(self):
        ratios = [0.0]
        with pytest.raises(ValueError):
            get_ratio_list_sum(ratios)

    def test_invalid_ratios_negative(self):
        ratios = [-0.1, 0.5]
        with pytest.raises(ValueError, match="Ratios must be non-negative."):
            get_ratio_list_sum(ratios)

    def test_invalid_ratios_sum_greater_than_one(self):
        ratios = [0.6, 0.6]
        with pytest.raises(ValueError, match="Sum of ratios must be 1.0."):
            get_ratio_list_sum(ratios)


class TestGetRatiosForCounts:
    def test_simple_counts(self):
        counts = [10, 20, 30]
        ratios = get_ratios_for_counts(counts)
        # First n-1 ratios are directly proportional
        assert ratios[:-1] == [10 / 60, 20 / 60]
        assert pytest.approx(sum(ratios), rel=1e-12) == 1.0

    def test_single_count_gets_full_ratio(self):
        counts = [42]
        ratios = get_ratios_for_counts(counts)
        assert ratios == [1.0]

    def test_empty_counts_raises_value_error(self):
        counts = []
        with pytest.raises(ValueError, match="Counts list cannot be empty"):
            get_ratios_for_counts(counts)

    def test_zero_total_counts_raises_value_error(self):
        counts = [0, 0, 0]
        with pytest.raises(ValueError, match="Total count must be greater than 0"):
            get_ratios_for_counts(counts)

    def test_last_ratio_adjustment_sums_to_one(self):
        counts = [1, 2, 3]
        ratios = get_ratios_for_counts(counts)
        total = sum(counts)
        assert ratios[:-1] == [c / total for c in counts[:-1]]
        assert pytest.approx(sum(ratios), rel=1e-12, abs=1e-12) == 1.0
        assert pytest.approx(ratios[-1], rel=1e-12, abs=1e-12) == 1.0 - sum(ratios[:-1])


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
            "embeddings": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
            ),
        }
        to_remove = {1, 3}
        result = filter_batch_from_indices(batch, to_remove)
        assert torch.equal(result["idx"], torch.tensor([0, 2]))
        assert torch.equal(result["embeddings"], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

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


class TestGetIndicesWithExcludedLabels:
    def test_returns_indices_for_excluded_labels(self):
        batch = {
            "idx": [0, 1, 2, 3, 4],
            "label": ["cat", "dog", "cat", "bird", "dog"],
        }
        result = get_indices_with_labels(batch, "label", {"cat", "dog"})
        assert result == {0, 1, 2, 4}

    def test_returns_empty_set_when_no_match(self):
        batch = {
            "idx": [0, 1, 2],
            "label": ["cat", "cat", "cat"],
        }
        result = get_indices_with_labels(batch, "label", {"dog"})
        assert result == set()

    def test_returns_all_indices_when_all_excluded(self):
        batch = {
            "idx": [10, 20, 30],
            "label": ["a", "b", "a"],
        }
        result = get_indices_with_labels(batch, "label", {"a", "b"})
        assert result == {10, 20, 30}

    def test_with_torch_tensor_indices(self):
        batch = {
            "idx": torch.tensor([5, 6, 7, 8]),
            "label": ["keep", "exclude", "keep", "exclude"],
        }
        result = get_indices_with_labels(batch, "label", {"exclude"})
        assert result == {6, 8}

    def test_custom_index_key(self):
        batch = {
            "idx_vector": [100, 200, 300],
            "label": ["a", "b", "a"],
        }
        result = get_indices_with_labels(batch, "label", {"a"}, index_key="idx_vector")
        assert result == {100, 300}

    def test_label_values_as_list(self):
        batch = {
            "idx": [0, 1, 2, 3],
            "label": [["cat"], ["dog", "mouse"], ["bird"], ["cat", "dog"]],
        }
        result = get_indices_with_labels(batch, "label", {"cat", "dog"})
        assert result == {0, 1, 3}


class TestSplitSamplingAmongChunks:
    def test_single_chunk(self) -> None:
        result = split_sampling_among_chunks(10, [100])
        assert result == [10]

    def test_too_small_single_chunk(self) -> None:
        with pytest.raises(
            ValueError, match="Split count .* cannot be greater than chunk size"
        ):
            split_sampling_among_chunks(10, [9])

    def test_proportional_split_two_chunks(self) -> None:
        to_split = 10
        chunk_sizes = [30, 70]
        split = split_sampling_among_chunks(to_split, chunk_sizes)
        assert sum(split) == to_split
        assert split[0] == int((30 / sum(chunk_sizes)) * to_split)

    def test_proportional_split_multiple_chunks(self) -> None:
        to_split = 25
        chunk_sizes = [10, 20, 30]
        split = split_sampling_among_chunks(to_split, chunk_sizes)
        assert sum(split) == to_split
        assert len(split) == len(chunk_sizes)

    def test_empty_chunk_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one chunk size is required"):
            split_sampling_among_chunks(10, [])

    def test_zero_total_chunk_sizes_raise(self) -> None:
        with pytest.raises(
            ValueError, match="Total size of chunks must be greater than 0"
        ):
            split_sampling_among_chunks(10, [0, 0, 0])

    def test_split_count_exceeds_chunk_size_raises(self) -> None:
        to_split = 5
        chunk_sizes = [3]
        with pytest.raises(
            ValueError, match="Split count .* cannot be greater than chunk size"
        ):
            split_sampling_among_chunks(to_split, chunk_sizes)


class TestTransformBatchFromMultipleToBinarizedTargets:
    def test_simple_usage(self):
        target = torch.zeros(2, 2)
        target[0, 0] = 25
        data = torch.arange(6).reshape(2, 3)
        batch = {
            "data": data,
            "target": target,
            "label": [
                {"A", "B"},
                {
                    "B",
                },
            ],
        }

        target_to_label = {
            (0, 0): "A",
            (25, 0): "B",
        }

        out = transform_batch_from_multiple_to_binarized_targets(
            batch=batch,
            target_key="target",
            label_key="label",
            target_to_label=target_to_label,
        )

        assert (out["data"] == torch.cat([data] * 2, dim=0)).all()
        new_target = torch.zeros(4, 1)
        new_target[1, 0] = 1
        new_target[2, 0] = 1
        assert (out["target"] == new_target).all()
        assert out["label"] == ["A", "A", "B", "B"]

    def test_with_merge_multilabel(self):
        target = torch.zeros(2, 2)
        target[0, 0] = 25
        data = torch.arange(6).reshape(2, 3)
        batch = {
            "data": data,
            "target": target,
            "label": [
                {"A", "B"},
                {
                    "B",
                },
            ],
        }

        target_to_label = {
            (0, 0): "A",
            (25, 0): "B",
        }

        out = transform_batch_from_multiple_to_binarized_targets(
            batch=batch,
            target_key="target",
            label_key="label",
            target_to_label=target_to_label,
            merge_labels=True,
        )

        assert (out["data"] == data).all()
        new_target = torch.ones(2, 1)
        assert (out["target"] == new_target).all()
        assert out["label"] == ["A_B", "A_B"]

    def test_with_exclude_zero(self):
        target = torch.zeros(2, 1)
        target[0, 0] = 25
        data = torch.arange(6).reshape(2, 3)
        batch = {
            "data": data,
            "target": target,
            "label": [
                {
                    "A",
                },
                {
                    "B",
                },
            ],
        }

        target_to_label = {
            (0,): "A",
            (25,): "B",
        }

        out = transform_batch_from_multiple_to_binarized_targets(
            batch=batch,
            target_key="target",
            label_key="label",
            target_to_label=target_to_label,
            exclude_full_zero=True,
        )

        assert (out["data"] == data).all()
        new_target = torch.zeros(2, 1)
        new_target[0, 0] = 1
        assert (out["target"] == new_target).all()
        assert set(out["label"]) == {
            "B",
        }

    def test_with_exclude_labels(self):
        target = torch.zeros(2, 1)
        target[0, 0] = 25
        data = torch.arange(6).reshape(2, 3)
        batch = {
            "data": data,
            "target": target,
            "label": [
                {
                    "A",
                },
                {
                    "B",
                },
            ],
        }

        target_to_label = {
            (0,): "A",
            (25,): "B",
        }

        out = transform_batch_from_multiple_to_binarized_targets(
            batch=batch,
            target_key="target",
            label_key="label",
            target_to_label=target_to_label,
            exclude_labels={"A"},
        )

        assert (out["data"] == torch.cat([data], dim=0)).all()
        new_target = torch.zeros(2, 1)
        new_target[0, 0] = 1
        assert (out["target"] == new_target).all()
        assert set(out["label"]) == {"B"}

    def test_with_target_to_label_missing_failure(self):
        target = torch.zeros(2, 1)
        target[0, 0] = 25
        data = torch.arange(6).reshape(2, 3)
        batch = {
            "data": data,
            "target": target,
            "label": [
                {
                    "A",
                },
                {
                    "B",
                },
            ],
        }

        target_to_label = {
            (1, 0): "A",
        }

        with pytest.raises(
            ValueError, match="Unique target .* not found in target_to_label mapping"
        ):
            transform_batch_from_multiple_to_binarized_targets(
                batch=batch,
                target_key="target",
                label_key="label",
                target_to_label=target_to_label,
            )

    def test_with_no_target_failure(self):
        target = torch.zeros(2, 1)
        data = torch.arange(6).reshape(2, 3)
        batch = {
            "data": data,
            "target": target,
            "label": [
                {
                    "A",
                },
                {
                    "B",
                },
            ],
        }

        target_to_label = {
            (1, 0): "A",
        }

        with pytest.raises(
            ValueError,
            match="No valid targets found after applying exclude_full_zero filter.",
        ):
            transform_batch_from_multiple_to_binarized_targets(
                batch=batch,
                target_key="target",
                label_key="label",
                target_to_label=target_to_label,
                exclude_full_zero=True,
            )


class TestGetSamplingCountFromRatios:
    def test_simple_usage(self) -> None:
        ratios = {"a": 0.5, "b": 0.3, "c": 0.2}
        sampling_size = 10

        counts = get_sampling_count_from_ratios(ratios, sampling_size)

        assert counts == {"a": 5, "b": 3, "c": 2}

    def test_with_distribution(self) -> None:
        ratios = {"a": 0.5, "b": 0.45, "c": 0.05}
        sampling_size = 11

        counts = get_sampling_count_from_ratios(ratios, sampling_size)

        assert counts == {"a": 6, "b": 5, "c": 0}

    def test_with_force_non_zero(self) -> None:
        ratios = {"a": 0.5, "b": 0.4, "c": 0.1}
        sampling_size = 5

        counts = get_sampling_count_from_ratios(ratios, sampling_size, True)

        assert counts == {"a": 2, "b": 2, "c": 1}

    def test_with_force_non_zero_with_highest_0_failure(self) -> None:
        ratios = {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.199, "f": 0.001}
        sampling_size = 5

        with pytest.raises(ValueError, match="While trying to adjust counts"):
            get_sampling_count_from_ratios(ratios, sampling_size, True)

    def test_ratios_sum_not_one_failure(self) -> None:
        ratios = {
            "k1": 0.5,
            "k2": 0.49,
        }
        sampling_size = 3

        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            get_sampling_count_from_ratios(
                ratios,
                sampling_size,
            )


class TestTransformBatchFromMultilabelToBinaryLabels:
    def test_single_string_labels(self) -> None:
        batch: dict[str, torch.Tensor | list] = {
            "x": torch.arange(4).reshape(2, 2),
            "label": ["a", "b"],
        }

        out = transform_batch_from_multilabel_to_independent_labels(
            batch=batch,
            label_key="label",
        )
        x_out = out["x"]
        x_batch = batch["x"]
        assert isinstance(x_out, torch.Tensor)
        assert isinstance(x_batch, torch.Tensor)
        assert torch.equal(x_out, x_batch)
        assert out["label"] == ["a", "b"]

    def test_multilabel_sets(self) -> None:
        x = torch.arange(6).reshape(3, 2)
        batch: dict[str, torch.Tensor | list] = {
            "x": x,
            "label": [
                {"a", "b"},
                {"b"},
                {"c", "a"},
            ],
        }

        out = transform_batch_from_multilabel_to_independent_labels(
            batch=batch,
            label_key="label",
        )

        expected_x = torch.stack([x[0], x[0], x[1], x[2], x[2]])
        x_out = out["x"]
        assert isinstance(x_out, torch.Tensor)
        assert torch.equal(x_out, expected_x)
        expected_labels = sorted(["a", "b", "b", "c", "a"])
        assert sorted(out["label"]) == expected_labels

    def test_with_merge_labels(self) -> None:
        x = torch.arange(6).reshape(3, 2)
        batch: dict[str, torch.Tensor | list] = {
            "x": x,
            "label": [
                {"a", "b"},
                {"b"},
                {"c", "a"},
            ],
        }

        out = transform_batch_from_multilabel_to_independent_labels(
            batch=batch,
            label_key="label",
            merge_labels=True,
        )

        x_out = out["x"]
        assert isinstance(x_out, torch.Tensor)
        assert torch.equal(x_out, x)
        expected_labels = sorted(
            [
                "a_b",
                "a_c",
                "b",
            ]
        )
        assert sorted(out["label"]) == expected_labels

    def test_exclude_labels(self) -> None:
        x = torch.arange(6).reshape(3, 2)
        batch: dict[str, list | torch.Tensor] = {
            "x": x,
            "label": [
                {"a", "b"},
                {"b"},
                {"c", "a"},
            ],
            "other": ["keep1", "keep2", "keep3"],
        }

        out = transform_batch_from_multilabel_to_independent_labels(
            batch=batch,
            label_key="label",
            exclude_labels={"b"},
        )

        expected_labels = sorted(["a", "c", "a"])
        expected_x = torch.stack([x[0], x[2], x[2]])
        expected_other = ["keep1", "keep3", "keep3"]
        x_out = out["x"]
        assert isinstance(x_out, torch.Tensor)
        assert torch.equal(x_out, expected_x)
        assert sorted(out["label"]) == expected_labels
        assert out["other"] == expected_other
