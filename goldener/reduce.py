import torch

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

from goldener.torch_utils import torch_tensor_to_numpy_vectors, np_transform_from_torch


class GoldReducer:
    """Dimensionality reduction using UMAP, PCA, TSNE, or GaussianRandomProjection.

    This reducer only accepts already vectorized input as 2D torch.Tensor where each row
    represents a data point and each column represents a feature. The input must have
    shape (batch_size, feature_dim).

    Raw data (images, text, etc.) must be vectorized using appropriate tools (e.g., GoldVectorizer)
    before being passed to this reducer.

    Attributes:
        reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection.
    """

    def __init__(self, reducer: UMAP | PCA | TSNE | GaussianRandomProjection):
        """Initialize the GoldReducer.

        Args:
            reducer: An instance of UMAP, PCA, TSNE, or GaussianRandomProjection for dimensionality reduction.
        """
        self.reducer = reducer

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate that input is already vectorized (2D torch.Tensor).

        Args:
            x: Input tensor to validate.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        if x.ndim != 2:
            raise ValueError(
                f"GoldReducer only accepts 2D tensors (batch_size, feature_dim). "
                f"Got shape {x.shape}. Please ensure your input is already vectorized."
            )

    def fit(self, x: torch.Tensor) -> None:
        """Fit the dimensionality reduction model to the data.

        Args:
            x: Already vectorized 2D input tensor of shape (batch_size, feature_dim) to fit the model on.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        self._validate_input(x)
        x_np = torch_tensor_to_numpy_vectors(x)
        self.reducer.fit(x_np)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the dimensionality reduction model to the data and transform it.

        Args:
            x: Already vectorized 2D input tensor of shape (batch_size, feature_dim) to fit and transform.

        Returns:
            Transformed tensor with reduced dimensionality.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        self._validate_input(x)
        return np_transform_from_torch(x, self.reducer.fit_transform)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the data using the fitted dimensionality reduction model.

        Args:
            x: Already vectorized 2D input tensor of shape (batch_size, feature_dim) to transform.

        Returns:
            Transformed tensor with reduced dimensionality.

        Raises:
            ValueError: If input is not a 2D tensor.
        """
        self._validate_input(x)
        return np_transform_from_torch(x, self.reducer.transform)
