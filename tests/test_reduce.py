import torch
import pytest

from goldener.reduce import GoldReducer
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection


def test_pca():
    data = torch.randn(10, 5, dtype=torch.float32)
    reducer = PCA(n_components=3)
    dr = GoldReducer(reducer)
    # Test fit + transform
    dr.fit(data)
    out = dr.transform(data)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (data.shape[0], 3)
    # Test fit_transform
    out2 = dr.fit_transform(data)
    assert isinstance(out2, torch.Tensor)
    assert out2.shape == (data.shape[0], 3)


def test_tsne():
    data = torch.randn(10, 5, dtype=torch.float32)
    reducer = TSNE(n_components=2, perplexity=3)
    dr = GoldReducer(reducer)
    # TSNE does not have transform, only fit_transform
    out = dr.fit_transform(data)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (data.shape[0], 2)


def test_umap():
    data = torch.randn(10, 5, dtype=torch.float32)
    reducer = UMAP(n_components=2)
    dr = GoldReducer(reducer)
    # Test fit + transform
    dr.fit(data)
    out = dr.transform(data)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (data.shape[0], 2)
    # Test fit_transform
    out2 = dr.fit_transform(data)
    assert isinstance(out2, torch.Tensor)
    assert out2.shape == (data.shape[0], 2)


def test_gaussian_random_projection():
    data = torch.randn(10, 5, dtype=torch.float32)
    reducer = GaussianRandomProjection(n_components=4)
    dr = GoldReducer(reducer)
    # Test fit + transform
    dr.fit(data)
    out = dr.transform(data)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (data.shape[0], 4)
    # Test fit_transform
    out2 = dr.fit_transform(data)
    assert isinstance(out2, torch.Tensor)
    assert out2.shape == (data.shape[0], 4)


class TestGoldReducer:
    def test_reducer_rejects_0d_tensor(self):
        """Test that reducer rejects 0D tensor input and requires 2D tensor."""
        data_0d = torch.tensor(5.0, dtype=torch.float32)
        reducer = PCA(n_components=3)
        dr = GoldReducer(reducer)

        # fit should reject 0D tensor
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.fit(data_0d)

        # fit_transform should reject 0D tensor
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.fit_transform(data_0d)

        # transform should reject 0D tensor
        dr.fit(torch.randn(10, 5))  # First fit with valid input
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.transform(data_0d)

    def test_reducer_rejects_1d_tensor(self):
        """Test that reducer rejects 1D tensor input and requires 2D tensor."""
        data_1d = torch.randn(10, dtype=torch.float32)
        reducer = PCA(n_components=3)
        dr = GoldReducer(reducer)

        # fit should reject 1D tensor
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.fit(data_1d)

        # fit_transform should reject 1D tensor
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.fit_transform(data_1d)

        # transform should reject 1D tensor
        dr.fit(torch.randn(10, 5))  # First fit with valid input
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.transform(data_1d)

    def test_reducer_rejects_3d_tensor(self):
        """Test that reducer rejects 3D tensor input and requires 2D tensor."""
        data_3d = torch.randn(10, 5, 3, dtype=torch.float32)
        reducer = PCA(n_components=3)
        dr = GoldReducer(reducer)

        # fit should reject 3D tensor
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.fit(data_3d)

        # fit_transform should reject 3D tensor
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.fit_transform(data_3d)

        # transform should reject 3D tensor
        dr.fit(torch.randn(10, 5))  # First fit with valid input
        with pytest.raises(ValueError, match="GoldReducer only accepts 2D tensors"):
            dr.transform(data_3d)

    def test_reducer_accepts_valid_2d_tensor(self):
        """Test that reducer correctly accepts valid 2D tensor input."""
        data = torch.randn(10, 5, dtype=torch.float32)
        reducer = PCA(n_components=3)
        dr = GoldReducer(reducer)

        # Should work without errors
        dr.fit(data)
        out = dr.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (10, 3)

        # fit_transform should also work
        out2 = dr.fit_transform(data)
        assert isinstance(out2, torch.Tensor)
        assert out2.shape == (10, 3)
