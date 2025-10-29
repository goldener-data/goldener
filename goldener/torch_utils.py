from typing import Callable

import numpy as np
import torch


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
    return x.cpu().numpy()


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
        _data_iterator: The current iterator over the dataset.
    """

    def __init__(self, data_iterable: torch.utils.data.IterableDataset):
        super().__init__()
        self.data_iterable = data_iterable
        self._data_iterator = iter(self.data_iterable)

    def __iter__(self):
        if self._data_iterator is None:
            self._data_iterator = iter(self.data_iterable)
        return self

    def __next__(self):
        try:
            return next(self._data_iterator)
        except StopIteration:
            self._data_iterator = None
            raise

    def reset(self):
        self._data_iterator = iter(self.data_iterable)
