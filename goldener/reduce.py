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
        self.reducer = reducer

    def fit(self, x: torch.Tensor) -> None:
        """Fit the dimensionality reduction model to the data."""
        x_np = torch_tensor_to_numpy_vectors(x)
        self.reducer.fit(x_np)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the dimensionality reduction model to the data."""
        return np_transform_from_torch(x, self.reducer.fit_transform)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the data using the fitted dimensionality reduction model."""
        return np_transform_from_torch(x, self.reducer.transform)
