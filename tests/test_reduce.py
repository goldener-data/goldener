import torch
from torch.nn import Conv2d, Linear

from goldener.reduce import GoldSKLearnReductionTool, GoldTorchModuleReductionTool
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection


class TestGoldSKLearnReductionTool:
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
    def test_torch_module_reduction_linear(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        linear = Linear(in_features=5, out_features=3)
        rd = GoldTorchModuleReductionTool(linear)
        out = rd.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (data.shape[0], 3)
        assert out.dtype == data.dtype

    def test_torch_module_reduction_conv2d(self):
        data = torch.randn(4, 3, 8, 8, dtype=torch.float32)
        conv = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=0, stride=1)
        rd = GoldTorchModuleReductionTool(conv)
        out = rd.transform(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 6, 6, 6)
        assert out.dtype == data.dtype
