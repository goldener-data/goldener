from collections.abc import Sequence
from logging import getLogger
from typing import Callable, Any, Iterator, TypeVar

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset, default_collate


logger = getLogger(__name__)
T = TypeVar("T")


def collate_keeping_sequences_as_sequences(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate dictionary samples while preserving sequence-valued features.

    Args:
        batch: A list of samples, where each sample is a dictionary.

    Returns:
        A dictionary with sequences kept per sample and all other values collated
        with PyTorch's default collate function.

    Raises:
        KeyError: If a sample is missing a key present in the first sample.
    """
    if not batch:
        return {}

    values_by_key = {key: [value] for key, value in batch[0].items()}
    for sample_idx, sample in enumerate(batch[1:], start=1):
        for key in values_by_key:
            values_by_key[key].append(sample[key])

        extra_keys = sample.keys() - values_by_key.keys()
        if extra_keys:
            logger.warning(
                "Ignoring extra keys %s in batch sample at index %d.",
                sorted(extra_keys),
                sample_idx,
            )

    return {
        key: (
            values
            if isinstance(values[0], Sequence) or values[0] is None
            else default_collate(values)
        )
        for key, values in values_by_key.items()
    }


def make_2d_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert a torch tensor to a 2D tensor.

    If the input tensor is 0D, it is converted to a (1, 1) tensor.
    If the input tensor is 1D, it is converted to a (N, 1) tensor.
    If the input tensor is 2D, it is kept as (N, M) tensor.
    If the input tensor is 3D or higher, the second dimension is moved to the last
    dimension, and the first dimensions are flattened to form a 2D tensor.

    Args:
        x: Input tensor convert to 2D tensor.

    Returns:
        A 2D tensor of shape (N, M)
    """
    initial_ndim = x.ndim
    if initial_ndim == 0:
        x = x.unsqueeze(0).unsqueeze(0)
    elif initial_ndim == 1:
        x = x.unsqueeze(1)

    if x.ndim == 2:
        return x

    x = x.moveaxis(1, -1)
    return x.reshape(-1, x.shape[-1])


def torch_tensor_to_numpy_vectors(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array of vectors.

    If the input tensor is 0D, it is converted to a (1, 1) array.
    If the input tensor is 1D, it is converted to a (N, 1) array.
    If the input tensor is 2D, it is kept as (N, M) array.
    If the input tensor is 3D or higher, the second dimension is moved to the last
    dimension, and the first dimensions are flattened to form a 2D array of vectors.

    Args:
        x: Input tensor convert to numpy vectors.

    Returns:
        A 2D numpy array of shape (N, M)
    """
    x = make_2d_tensor(x)
    return x.detach().cpu().numpy()


def numpy_vectors_to_torch_tensor(
    x: np.ndarray, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Convert a numpy array to a torch tensor.

    If the input array is 0D, it is converted to a (1, 1) tensor.
    If the input array is 1D, it is converted to a (N, 1) tensor.
    If the input array is 2D, it is kept as (N, M) tensor.
    If the input array is 3D or higher, the last dimension is moved to the second
    dimension, and the first dimensions are reshaped to form the desired shape.

    Args:
        x: Input numpy array to convert to torch tensor.
        shape: Desired shape of the output tensor.
        dtype: Desired data type of the output tensor.
        device: Desired device of the output tensor.

    Returns:
        A torch tensor of the desired shape, dtype, and device.

    Raises:
        ValueError: If the desired shape is less than 2D.
    """
    if len(shape) < 2:
        raise ValueError("Shape must be at least 2D")

    if x.ndim <= 2:
        return (
            torch.from_numpy(x)
            .to(
                device=device,
                dtype=dtype,
            )
            .reshape(shape)
        )

    x_torch = torch.from_numpy(x)
    x_torch = x_torch.moveaxis(-1, 1)
    x_torch = x_torch.reshape(shape)

    return x_torch.to(
        device=device,
        dtype=dtype,
    )


def np_transform_from_torch(
    x: torch.Tensor,
    transform_np: Callable[[np.ndarray], np.ndarray],
) -> torch.Tensor:
    """Apply a numpy transformation to a torch tensor.

    Args:
        x: Input tensor to transform.
        transform_np: A callable that takes a numpy array and returns a transformed
            numpy array.

    Returns:
        A transformed torch tensor with the same dtype and device as the input tensor.
    """
    x_np = torch_tensor_to_numpy_vectors(x)
    transformed = transform_np(x_np)
    return numpy_vectors_to_torch_tensor(
        transformed, shape=transformed.shape, dtype=x.dtype, device=x.device
    )


class ResetableTorchIterableDataset(torch.utils.data.IterableDataset):
    """A resetable iterable dataset that can be re-initialized.

    It is useful for accessing the first element (to set up a table for instance)
    of the dataset before looping over the full dataset.

    Attributes:
        data_iterable: The underlying iterable dataset.
        _data_iterator: The current iterator over the dataset, or None once it
            has been exhausted (see `__next__`) and not yet re-created.
    """

    def __init__(self, data_iterable: torch.utils.data.IterableDataset):
        """Initialize the resetable iterable dataset.

        Args:
            data_iterable: The underlying iterable dataset to wrap.
        """
        super().__init__()
        self.data_iterable = data_iterable
        self._data_iterator: Iterator[Any] | None = iter(self.data_iterable)

    def __iter__(self):
        """Return the iterator object."""
        if self._data_iterator is None:
            self._data_iterator = iter(self.data_iterable)
        return self

    def __next__(self):
        """Return the next item from the iterator."""
        # __iter__ always runs before __next__ in the iterator protocol and
        # re-creates _data_iterator if it was exhausted, so it is never None
        # here at runtime; the assert documents that invariant for mypy.
        assert self._data_iterator is not None
        try:
            return next(self._data_iterator)
        except StopIteration:
            self._data_iterator = None
            raise

    def reset(self):
        """Reset the iterator to the beginning of the dataset."""
        self._data_iterator = iter(self.data_iterable)


def get_dataset_sample_dict(
    dataset: Dataset,
    collate_fn: Callable | None = None,
    expected: list[str] | None = None,
    excluded: list[str] | None = None,
) -> dict[str, Any]:
    """Get a sample from a dataset as a dictionary and validate some of its keys.

    Args:
        dataset: The dataset to get a sample from.
        collate_fn: An optional collate function to apply to the sample.
        expected: An optional list of keys that must be present in the sample.
        excluded: An optional list of keys that must not be present in the sample.

    Returns:
        A sample from the dataset as a dictionary.

    Raises:
        ValueError: If the sample is not a dictionary after applying the `collate_fn`.
        ValueError: If the sample is missing any of the `expected` keys.
        ValueError: If the sample contains any of the `excluded` keys.
    """

    if isinstance(dataset, IterableDataset):
        dataset = ResetableTorchIterableDataset(dataset)

    sample = (
        next(dataset)
        if isinstance(dataset, ResetableTorchIterableDataset)
        else dataset[0]
    )
    if isinstance(dataset, ResetableTorchIterableDataset):
        dataset.reset()

    if collate_fn is not None:
        sample = collate_fn([sample])

    if not isinstance(sample, dict):
        raise ValueError("Sample must be a dictionary after applying the collate_fn.")

    if expected is not None:
        not_present_keys = [key for key in expected if key not in sample]
        if len(not_present_keys) > 0:
            raise ValueError(f"Sample is missing expected keys: {not_present_keys}")

    if excluded is not None:
        present_rejected_keys = [key for key in excluded if key in sample]
        if len(present_rejected_keys) > 0:
            raise ValueError(f"Sample contains rejected keys: {present_rejected_keys}")

    return sample


def get_unique_values_in_tensor(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Get the unique values/vectors of a given dimension in a tensor.

    Args:
        tensor: The input tensor to get unique values from.
        dim: The dimension along which to get unique values. Default is 1.

    Returns:
        A tensor with all the unique values of the input tensor for teh specified dimension.
    """
    return torch.unique(tensor.movedim(dim, -1).reshape(-1, tensor.shape[dim]), dim=0)


def shuffle_list(items: list[T], generator: torch.Generator) -> list[T]:
    """Shuffle the given items in a list.

    Args:
        items: The list of items to shuffle.
        generator: A torch generator to manage the random shuffling.

    Returns: A list of shuffled items.
    """
    return [
        items[int(i)] for i in torch.randperm(len(items), generator=generator).tolist()
    ]
