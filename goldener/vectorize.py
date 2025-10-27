from dataclasses import dataclass
from functools import partial

from torch.utils.data import RandomSampler
from typing_extensions import assert_never
from typing import Callable

from enum import Enum

import torch
from torch import Generator

from goldener.torch_utils import make_2d_tensor
from goldener.utils import check_x_and_y_shapes


class FilterLocation(Enum):
    START = "start"
    END = "end"
    RANDOM = "random"


class Filter2DWithCount:
    """Filter 2D tensor rows based on specified criteria.

    Attributes:
        _filter_location: Location to filter from (start, end, random).
        _filter_count: Number of rows to filter.
        _keep: Whether to keep or remove the filtered rows.
        _random_sampler: Sampler for random filtering.
    """

    def __init__(
        self,
        filter_count: int = 1,
        filter_location: FilterLocation = FilterLocation.RANDOM,
        keep: bool = False,
        generator: Generator | None = None,
    ) -> None:
        """Initialize the Filter2DWithCount.

        Args:
            filter_count: Number of rows to filter.
            filter_location: Location to filter from (start, end, random).
            keep: Whether to keep or remove the filtered rows.
            generator: Random number generator for random filtering.
        """
        if filter_count <= 0:
            raise ValueError("filter_count must be greater than 0")

        self._filter_location = filter_location
        self._filter_count = filter_count
        self._keep = keep

        if self._filter_location == FilterLocation.RANDOM:
            self._random_sampler = partial(
                RandomSampler,
                replacement=False,
                num_samples=self._filter_count,
                generator=generator,
            )

    @property
    def is_random(self):
        """Check if the filter is random."""
        return self._filter_location is FilterLocation.RANDOM

    def filter(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.filter_tensors({"x": x})["x"]

    def filter_tensors(
        self,
        x: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Filter the input tensor or dictionary of tensors.

        Args:
            x: Input 2D tensor or dictionary of 2D tensors to filter. In case of a dictionary,
            all tensors must have the same number of rows and this number must be greater than filter_count.

        Returns:
            Filtered tensor or dictionary of tensors. If the batch size is less than filter_count,
            the input is returned unchanged.

        Raises:
            ValueError: If input tensors do not have the same batch size or are not 2 dimensional.
        """
        first_tensor = next(iter(x.values()))
        batch_size = first_tensor.shape[0]
        for tensor in x.values():
            if tensor.ndim != 2:
                raise ValueError(
                    "All input tensors must be 2D to filter them with Filter2DWithCount."
                )

            if tensor.shape[0] != batch_size:
                raise ValueError(
                    "All input tensors must have the same batch size to filter them with Filter2DWithCount."
                )

        if batch_size < self._filter_count:
            return x

        if self._filter_location is FilterLocation.START:
            return self._start_filter(x)

        elif self._filter_location is FilterLocation.END:
            return self._end_filter(x)

        elif self._filter_location is FilterLocation.RANDOM:
            return self._random_filter(x)

        else:
            assert_never(self._filter_location)

    def _start_filter(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return (
            {k: v[: self._filter_count] for k, v in x.items()}
            if self._keep
            else {k: v[self._filter_count :] for k, v in x.items()}
        )

    def _end_filter(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return (
            {k: v[-self._filter_count :] for k, v in x.items()}
            if self._keep
            else {k: v[: -self._filter_count] for k, v in x.items()}
        )

    def _random_filter(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        first_value = next(iter(x.values()))
        indices = list(self._random_sampler(range(len(first_value))))
        if self._keep:
            return {k: v[indices] for k, v in x.items()}
        else:
            mask = torch.ones(len(first_value), dtype=torch.bool)
            mask[indices] = 0
            return {k: v[mask.bool(), :] for k, v in x.items()}


@dataclass
class Vectorized:
    """Dataclass to hold vectorized tensors and the corresponding batch indices.

    Attributes:
        vectors: 2D tensor of vectorized data.
        batch_indices: 1D tensor containing information about the origin of each vector.
    """

    vectors: torch.Tensor
    batch_indices: torch.Tensor


class GoldVectorizer:
    """Transform input as 2D tensor and filter based on target tensor.

    Attributes:
        keep: Filter2DWithCount instance to keep specific rows in the input `x` of `filter`.
        remove: Filter2DWithCount instance to remove specific rows in the input `x` of `filter`.
        random_filter: Random Filter2DWithCount instance to randomly filter vectors randomly after
        applying `keep`, `remove` and the target `y` on the input `x` of filter.
        transform_y: Optional callable to transform the target tensor before transforming it to 2D.
    """

    def __init__(
        self,
        keep: Filter2DWithCount | None = None,
        remove: Filter2DWithCount | None = None,
        random_filter: Filter2DWithCount | None = None,
        transform_y: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the Vectorizer."""
        self.transform_y = transform_y

        if keep is not None and keep.is_random:
            raise ValueError("The 'keep' filter cannot be random.")
        self.keep = keep

        if remove is not None and remove.is_random:
            raise ValueError("The 'remove' filter cannot be random.")
        self.remove = remove

        if random_filter is not None and not random_filter.is_random:
            raise ValueError("The 'random_filter' must be random.")
        self.random_filter = random_filter

    def vectorize(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> Vectorized:
        """Vectorize input tensor and filter based on target tensor.

        If y is not provided, only filtering and vectorization are performed. If y is provided,
        it is used to filter the vectorized x tensor.

        Args:
            x: Input tensor to vectorize.
            y: Optional target tensor to filter the input tensor.

        Returns:
            Vectorized and filtered input tensor with the information
            about which element of the batch it corresponds to.

        Raises:
            ValueError: If x and y shapes are incompatible. See check_x_and_y_shapes for details.
        """
        if x.ndim < 3:
            raise ValueError("Input tensor x to vectorize must be at least 3D.")

        if y is not None and self.transform_y is not None:
            y = self.transform_y(y)

        filtered_x = []
        filtered_batch_info = []
        for idx_sample, x_sample in enumerate(x):
            x_sample = x_sample.unsqueeze(0)
            x_sample = make_2d_tensor(x_sample)

            x_sample = self._apply_filter(
                self.keep.filter if self.keep is not None else None, x_sample
            )
            x_sample = self._apply_filter(
                self.remove.filter if self.remove is not None else None, x_sample
            )

            if y is not None:
                y_sample = y[idx_sample].unsqueeze(0)
                x_sample = self._filter_2d_tensors_from_y(x_sample, y_sample)

            filtered_x.append(x_sample)
            filtered_batch_info.append(
                torch.full_like(x_sample[:, 0], idx_sample, dtype=torch.long)
            )

        return Vectorized(
            torch.cat(filtered_x, dim=0), torch.cat(filtered_batch_info, dim=0)
        )

    def _apply_filter(
        self,
        filter: Callable[[torch.Tensor], torch.Tensor] | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if filter is None:
            return x

        return filter(x)

    def _filter_2d_tensors_from_y(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Filter 2D tensor rows based on target tensor.

        When y is not the same batch size as x, it is expanded to match x's batch size.

        Args:
            x: Input 2D tensor to filter. It can be the output of `make_2d_tensor`.
            y: Target tensor to filter the input tensor. It is still in its raw configuration.
            The transform_y callable is applied to y before transforming it to a 2D tensor.

        Returns:
            Filtered input tensor.

        Raises:
            ValueError: If x and y shapes (after 2D transformation) are incompatible.
            See check_x_and_y_shapes for details.
            ValueError: If y tensor after transform contains only zeros.
        """
        y = make_2d_tensor(y)

        y_shape = y.shape
        x_shape = x.shape
        check_x_and_y_shapes(x_shape, y_shape)

        if (y == 0).all():
            raise ValueError(
                "The y tensor after transform must contain at least one "
                "non-zero value to select vectors from x."
            )

        return x[y.bool().squeeze(-1)]
