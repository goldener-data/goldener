import logging
from abc import ABC, abstractmethod

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
            x: x: Already vectorized 2D input tensor of shape (num_vectors, feature_dim) to transform.

        Returns:
            Transformed tensor with reduced dimensionality.
        """
        self._validate_input(x)
        return self.reducer(x)


class GoldSKLearnReductionTool(GoldReductionToolWithFit):
    """Dimensionality reduction of 2D vectors using UMAP, PCA, TSNE, or GaussianRandomProjection.

    Attributes:
        _reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection.
    """

    def __init__(self, reducer: UMAP | PCA | TSNE | GaussianRandomProjection):
        """Initialize the GoldSKLearnReductionTool.

        Args:
            reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection for dimensionality reduction.
        """
        self._reducer = reducer
        self._has_transform = hasattr(reducer, "transform")

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
