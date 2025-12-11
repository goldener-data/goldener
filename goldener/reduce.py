import torch

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

from goldener.torch_utils import torch_tensor_to_numpy_vectors, np_transform_from_torch


class GoldReducer:
    """Dimensionality reduction using UMAP, PCA, TSNE, or GaussianRandomProjection.

    Attributes:
        reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection.
    """

    def __init__(self, reducer: UMAP | PCA | TSNE | GaussianRandomProjection):
        """Initialize the GoldReducer.

        Args:
            reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection for dimensionality reduction.
        """
        self.reducer = reducer

    def fit(self, x: torch.Tensor) -> None:
        """Fit the dimensionality reduction model to the data.

        Args:
            x: Input tensor to fit the model on.
        """
        x_np = torch_tensor_to_numpy_vectors(x)
        self.reducer.fit(x_np)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the dimensionality reduction model to the data and transform it.

        Args:
            x: Input tensor to fit and transform.

        Returns:
            Transformed tensor with reduced dimensionality.
        """
        return np_transform_from_torch(x, self.reducer.fit_transform)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the data using the fitted dimensionality reduction model.

        Args:
            x: Input tensor to transform.

        Returns:
            Transformed tensor with reduced dimensionality.
        """
        return np_transform_from_torch(x, self.reducer.transform)
