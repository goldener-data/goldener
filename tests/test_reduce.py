import torch

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
