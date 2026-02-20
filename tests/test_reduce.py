import pytest
import torch
from torch.nn import Linear

from goldener.reduce import GoldSKLearnReductionTool, GoldTorchModuleReductionTool
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection


class TestGoldSKLearnReductionTool:
    def test_reducer_rejects_invalid(self):
        """Test that reducer rejects 1D tensor input and requires 2D tensor."""
        data_1d = torch.randn(10, dtype=torch.float32)
        reducer = PCA(n_components=3)
        dr = GoldSKLearnReductionTool(reducer)

        with pytest.raises(
            ValueError, match="GoldReductionTool only accepts 2D tensors"
        ):
            dr.fit(data_1d)

        with pytest.raises(
            ValueError, match="GoldReductionTool only accepts 2D tensors"
        ):
            dr.fit_transform(data_1d)

        dr.fit(torch.randn(10, 5))
        with pytest.raises(
            ValueError, match="GoldReductionTool only accepts 2D tensors"
        ):
            dr.transform(data_1d)

    def test_pca(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        reducer = PCA(n_components=3)
        rd = GoldSKLearnReductionTool(reducer)
        rd.fit(data)
        out = rd.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (data.shape[0], 3)
        out2 = rd.fit_transform(data)
        assert isinstance(out2, torch.Tensor)
        assert out2.shape == (data.shape[0], 3)

    def test_tsne(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        reducer = TSNE(n_components=2, perplexity=3)
        rd = GoldSKLearnReductionTool(reducer)
        rd.fit(data)
        out = rd.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (data.shape[0], 2)

        out = rd.fit_transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (data.shape[0], 2)

    def test_umap(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        reducer = UMAP(n_components=2)
        rd = GoldSKLearnReductionTool(reducer)
        rd.fit(data)
        out = rd.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (data.shape[0], 2)
        out2 = rd.fit_transform(data)
        assert isinstance(out2, torch.Tensor)
        assert out2.shape == (data.shape[0], 2)

    def test_gaussian_random_projection(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        reducer = GaussianRandomProjection(n_components=4)
        rd = GoldSKLearnReductionTool(reducer)
        rd.fit(data)
        out = rd.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (data.shape[0], 4)
        out2 = rd.fit_transform(data)
        assert isinstance(out2, torch.Tensor)
        assert out2.shape == (data.shape[0], 4)


class TestGoldTorchModuleReductionTool:
    def test_reducer_rejects_invalid(self):
        data_1d = torch.randn(10, dtype=torch.float32)
        reducer = Linear(in_features=5, out_features=3)
        dr = GoldTorchModuleReductionTool(reducer)

        with pytest.raises(
            ValueError, match="GoldReductionTool only accepts 2D tensors"
        ):
            dr.transform(data_1d)

    def test_torch_module_reduction_linear(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        linear = Linear(in_features=5, out_features=3)
        rd = GoldTorchModuleReductionTool(linear)
        out = rd.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (data.shape[0], 3)
        assert out.dtype == data.dtype
