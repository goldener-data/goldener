"""Dimensionality reduction of vectors.

The reduction tools reduce the dimension of 2D vectors, for instance to speed up the clustering or the selection
on high dimensional embeddings:

- `GoldReductionTool` is the abstract interface of all reduction tools.
- `GoldReductionToolWithFit` is the interface for the reduction methods requiring to be fitted on data.
- `GoldSKLearnReductionTool` wraps UMAP and scikit-learn methods (PCA, TSNE, GaussianRandomProjection).
- `GoldTorchModuleReductionTool` uses any PyTorch module to reduce the vectors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch.nn import Module

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection


logger = logging.getLogger(__name__)


class GoldReductionTool(ABC):
    """Reduce the dimensionality of 2D vectors.

    Base class for all dimensionality reduction tools.

    """

    @property
    def supports_transform(self) -> bool:
        """Whether this reducer can transform new rows using a fitted mapping."""
        return True

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate that input is already vectorized (2D torch.Tensor).

        Args:
            x: Input tensor to validate.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        if x.ndim != 2:
            raise ValueError(
                f"GoldReductionTool only accepts 2D tensors (num_vectors, feature_dim). "
                f"Got shape {x.shape}. Please ensure your input is already vectorized."
            )

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the 2D vectors using the dimensionality reduction tool.

        Args:
            x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to transform.

        Returns:
            Transformed tensor with reduced dimensionality.
        """


class GoldReductionToolWithFit(GoldReductionTool):
    """Reduce the dimensionality of 2D vectors from fittable reduction methods.

    Base class for dimensionality reduction tools that require fitting.
    """

    @abstractmethod
    def fit(self, x: torch.Tensor) -> None:
        """Fit the dimensionality reduction tool to the data.

        Args:
            x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to fit the model on.
        """

    @abstractmethod
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the dimensionality reduction tool to the 2D vectors and transform it.

        Args:
            x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to fit and transform.

        Returns:
            Transformed tensor with reduced dimensionality.
        """


class GoldTorchModuleReductionTool(GoldReductionTool):
    """Dimensionality reduction of 2D vectors with a torch Module.

    Attributes:
        reducer: The torch Module to use for dimensionality reduction.
    """

    def __init__(self, reducer: Module):
        """Initialize the GoldTorchModuleReductionTool.

        Args:
            reducer: The torch Module to use for dimensionality reduction.
        """
        self.reducer = reducer

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the 2D vectors using the fitted dimensionality reduction model.

        Args:
            x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to transform.

        Returns:
            Transformed tensor with reduced dimensionality.
        """
        self._validate_input(x)
        return self.reducer(x)


class GoldSKLearnReductionTool(GoldReductionToolWithFit):
    """Dimensionality reduction of 2D vectors using UMAP, PCA, TSNE, or GaussianRandomProjection.

    The randomness of the reducer (used by `fit`, `fit_transform` and, for reducers
    without `transform`, by the `transform` fallback) is driven by a NumPy
    `RandomState` whose state can be inspected with `random_state` and restored with
    `update_random_state` to replay the same random choices.

    Attributes:
        _reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection.
        _random_state: The `RandomState` shared with the reducer, or None when the
            reducer randomness is not controlled by this tool.
    """

    def __init__(
        self,
        reducer: UMAP | PCA | TSNE | GaussianRandomProjection,
        random_state: np.random.RandomState | int | None = None,
    ) -> None:
        """Initialize the GoldSKLearnReductionTool.

        Args:
            reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection for dimensionality reduction.
            random_state: Optional random state overriding the `random_state` of the reducer.
                An integer is used as seed of a new `RandomState`. When None, a `RandomState`
                already set on the reducer is reused; otherwise the reducer randomness is left
                untouched.
        """
        self._reducer = reducer
        self._has_transform = hasattr(reducer, "transform")

        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        elif random_state is None and isinstance(
            reducer.random_state, np.random.RandomState
        ):
            random_state = reducer.random_state

        self._random_state = random_state
        if random_state is not None:
            self._reducer.random_state = random_state

    @property
    def random_state(self) -> dict[str, Any] | None:
        """Return a snapshot of the reducer random state.

        Returns:
            An independent copy of the `RandomState` state, as returned by
            `numpy.random.RandomState.get_state(legacy=False)`, or None when the
            reducer randomness is not controlled by this tool.
        """
        if self._random_state is None:
            return None
        return self._random_state.get_state(legacy=False)

    @property
    def supports_transform(self) -> bool:
        """Whether the wrapped estimator has a fitted transform operation."""
        return self._has_transform

    def update_random_state(self, state: dict[str, Any]) -> None:
        """Restore the reducer random state.

        When the tool has no `RandomState` yet, a new one is created and set on the
        reducer, overriding its previous `random_state`.

        Args:
            state: Random state, as returned by `random_state`.
        """
        if self._random_state is None:
            self._random_state = np.random.RandomState()
            self._reducer.random_state = self._random_state
        self._random_state.set_state(state)

    def fit(self, x: torch.Tensor) -> None:
        """Fit the dimensionality reduction model to the 2D vectors.

        Args:
            x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to fit the model on.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        self._validate_input(x)
        x_np = x.detach().cpu().numpy()
        self._reducer.fit(x_np)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the dimensionality reduction model to the 2D vectors and transform it.

        Args:
            x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to fit and transform.

        Returns:
            Transformed tensor with reduced dimensionality.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        self._validate_input(x)
        x_np = x.detach().cpu().numpy()
        transformed = self._reducer.fit_transform(x_np)
        return torch.from_numpy(transformed).to(device=x.device, dtype=x.dtype)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the 2D vectors using the fitted dimensionality reduction model.

        If the estimator has no transform method, warn and fit it again on x
        using fit_transform.

        Args:
            x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to transform.

        Returns:
            Transformed tensor with reduced dimensionality.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        if not self._has_transform:
            logger.warning(
                "The provided reducer does not have a 'transform' method. "
                "Falling back to 'fit_transform' for transformation."
            )
            return self.fit_transform(x)

        self._validate_input(x)
        x_np = x.detach().cpu().numpy()
        transformed = self._reducer.transform(x_np)
        return torch.from_numpy(transformed).to(device=x.device, dtype=x.dtype)
