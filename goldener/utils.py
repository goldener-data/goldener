import math
from typing import Iterable, Any

import torch


def check_x_and_y_shapes(x_shape: tuple[int, ...], y_shape: tuple[int, ...]) -> None:
    """Check compatibility of shapes of x and y tensors.

    Args:
        x_shape: Shape of the input tensor x.
        y_shape: Shape of the target tensor y.

    Raises:
        ValueError: If the shapes are incompatible. See the conditions below for details.
    Conditions:
        - If x is 1D, x and y must have the same shape.
        - If x is 2D or more, y must have a channel dimension of 1 (i.e., y_shape[1] == 1).
        - If y has a batch size greater than 1 (i.e., y_shape[0] > 1), x and y must have the same batch size (i.e., x_shape[0] == y_shape[0]).
        - If x has more than 2 dimensions, the additional (after channel) dimensions of x and y must match.
    """

    if len(x_shape) == 1:
        if x_shape != y_shape:
            raise ValueError("x and y must have the same shape when x is 1D")
    else:
        if y_shape[1] != 1:
            raise ValueError(
                "x and y must have the same channel dimension when x is 2D"
            )

        batch_y = y_shape[0]
        if batch_y > 1 and x_shape[0] != batch_y:
            raise ValueError(
                "x and y must have the same batch size when y batch size > 1"
            )

        if len(x_shape) > 2 and x_shape[2:] != y_shape[2:]:
            raise ValueError(
                "x and y must have compatible shapes when x is more than 2D"
            )


def get_ratio_list_sum(ratios: list[float]) -> float:
    """Get the sum of a list of ratios and validate it (total between 0 and 1).

    Args:
        ratios: A list of float ratios.

    Returns:
        The sum of the ratios.

    Raises:
        ValueError: If the sum of the ratios is not greater than 0 and at most 1.
    """
    ratio_sum = sum(ratios)
    if not all(ratio >= 0 for ratio in ratios):
        raise ValueError("Ratios must be non-negative.")

    if not (0 < ratio_sum <= 1.0):
        raise ValueError("Sum of ratios must be 1.0.")

    return ratio_sum


def get_ratios_for_counts(counts: Iterable[int]) -> list[float]:
    """Get ratios for a list of counts.

    Args:
        counts: An iterable of integer counts.

    Returns:
        A list of float ratios corresponding to the input counts.
    """
    total = sum(counts)
    return [count / total for count in counts]


def filter_batch_from_indices(
    batch: dict[str, Any],
    to_remove: set[int],
    index_key: str = "idx",
) -> dict[str, Any]:
    """Filter a batch dictionary to only include items at specified indices.

    Args:
        batch: A dictionary representing a batch of data (each key corresponds to stacked information).
        to_remove: A set of indices to filter out from the batch.
        index_key: The key in the batch dictionary that contains the indices.

    Returns:
        A filtered batch dictionary without the specified indices.
    """
    keep_in_batch = [
        idx_position
        for idx_position, idx_value in enumerate(batch[index_key])
        if (idx_value.item() if isinstance(idx_value, torch.Tensor) else idx_value)
        not in to_remove
    ]
    if not keep_in_batch:
        return {}  # all samples already described

    def filter_batched_values(
        batched_value: list | torch.Tensor,
    ) -> list | torch.Tensor:
        """Inner function to remove already described samples from the batch."""
        filtered = [
            value
            for idx_value, value in enumerate(batched_value)
            if idx_value in keep_in_batch
        ]
        if isinstance(batched_value, torch.Tensor):
            return torch.stack(filtered, dim=0)
        else:
            return filtered

    return {
        key: filter_batched_values(batched_value)
        for key, batched_value in batch.items()
    }


def get_size_and_sampling_count_per_chunk(
    total_size: int, sampling_size: int, max_chunk_size: int
) -> tuple[list[int], list[int]]:
    """Get sizes and sampling sizes per chunk.

    Args:
        total_size: Total size of the data to sample from.
        sampling_size: Total number of samples to draw.
        max_chunk_size: Maximum size of each chunk.

    Returns:
        A tuple of two lists:
            - A list of integers representing the size of each chunk.
            - A list of integers representing the number of samples to draw from each chunk.

    Raises:
        ValueError: If sampling_size is greater than or equal to total_size.
    """
    if sampling_size >= total_size:
        raise ValueError("sampling_size must be less than or equal to total_size")

    if max_chunk_size >= total_size:
        return [total_size], [sampling_size]  # single chunk

    chunk_count = math.ceil(total_size / max_chunk_size)
    chunk_size = total_size // chunk_count
    chunk_sampling_size = sampling_size // chunk_count

    chunk_sizes = [chunk_size] * (chunk_count - 1)
    chunk_sizes.append(total_size - sum(chunk_sizes))
    chunk_sampling_sizes = [chunk_sampling_size] * (chunk_count - 1)
    chunk_sampling_sizes.append(sampling_size - sum(chunk_sampling_sizes))

    return chunk_sizes, chunk_sampling_sizes


def check_sampling_size(
    sampling_size: int | float, total_size: int | None = None
) -> None:
    """Check the validity of the sampling size.

    Args:
        sampling_size: The sampling size to check (can be int or float).
        total_size: The total size of the data (Optional).

    Raises:
        ValueError: If the sampling size is invalid based on its type and total size.
        If sampling_size is a float, it must be in the range (0.0, 1.0].
        If sampling_size is an int, it must be in the range (0, total_size].
    """
    if isinstance(sampling_size, float) and not (0 < sampling_size <= 1.0):
        raise ValueError(
            "Sampling size as float must be greater than 0.0 and at most 1.0"
        )

    if (
        isinstance(sampling_size, int)
        and total_size is not None
        and not (0 < sampling_size <= total_size)
    ):
        raise ValueError(
            "Sampling size as int must be greater than 0 and less or equal than the total number of samples"
        )


def check_all_same_type(iterable: Iterable[Any]) -> None:
    """Check if all elements in an iterable are of the same type.

    Args:
        iterable: An iterable containing elements to check.

    Raises:
        TypeError: If any element in the iterable is not of the expected type.
    """
    first_type = type(next(iter(iterable)))
    different = []
    for item in iterable:
        if not isinstance(item, first_type):
            different.append(type(item))

    if different:
        raise TypeError(
            f"All elements must be of the same type {first_type}. Found different types: {different}"
        )


def get_sampling_count_from_size(
    sampling_size: int | float, total_size: int | None = None
) -> int:
    """Get the sampling count from the sampling size.

    Args:
        sampling_size: The sampling size (can be int or float). If int, it represents the exact number of samples
        and it is required to be less than total_size. If float, it represents the fraction of total size and it
        is required to be in the range (0.0, 1.0].
        total_size: The total size of the data (Optional, required if sampling_size is float).

    Returns:
        The calculated sampling count as an integer.

    Raises:
        ValueError: If the sampling size is invalid based on its type and total size.
    """
    if isinstance(sampling_size, int):
        if sampling_size <= 0:
            raise ValueError("Sampling size as int must be greater than 0.")

        if total_size is not None and not (sampling_size < total_size):
            raise ValueError(
                "Sampling size as int must be less than the total number of samples."
            )

        return sampling_size

    if not (0 < sampling_size <= 1.0):
        raise ValueError(
            "Sampling size as float must be greater than 0.0 and at most 1.0."
        )

    if total_size is None:
        raise ValueError("Total size must be provided when sampling size is a float.")

    return math.ceil(sampling_size * total_size)
