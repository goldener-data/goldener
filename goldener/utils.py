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
    if not (0 < ratio_sum <= 1.0):
        raise ValueError("Sum of split ratios must be 1.0")

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
    batch: dict[str, Any], to_remove: set[int]
) -> dict[str, Any]:
    """Filter a batch dictionary to only include items at specified indices.

    Args:
        batch: A dictionary representing a batch of data (each key corresponds to stacked information).
        to_remove: A set of indices to filter out from the batch.

    Returns:
        A filtered batch dictionary without the specified indices.
    """
    keep_in_batch = [
        idx_position
        for idx_position, idx_value in enumerate(batch["idx"])
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
