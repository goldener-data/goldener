import numpy as np
import pytest
import torch
from torch.nn import Linear

from goldener.reduce import GoldSKLearnReductionTool, GoldTorchModuleReductionTool
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection


class TestGoldSKLearnReductionTool:
    @pytest.mark.parametrize(
        ("data", "message"),
        [
            pytest.param(torch.ones(10), "only accepts 2D tensors", id="1d"),
            pytest.param(torch.empty(2, 0), "positive feature width", id="zero-width"),
            pytest.param(
                torch.ones(2, 5, dtype=torch.int64), "finite floating", id="integer"
            ),
            pytest.param(torch.full((2, 5), float("nan")), "finite floating", id="nan"),
            pytest.param(
                torch.full((2, 5), float("inf")), "finite floating", id="infinity"
            ),
        ],
    )
    def test_reducer_rejects_invalid(self, data, message):
        reducer = PCA(n_components=3)
        dr = GoldSKLearnReductionTool(reducer)

        with pytest.raises(ValueError, match=message):
            dr.fit(data)

        with pytest.raises(ValueError, match=message):
            dr.fit_transform(data)

        dr.fit(torch.randn(10, 5))
        with pytest.raises(ValueError, match=message):
            dr.transform(data)

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

    def test_random_state_initialization(self):
        assert GoldSKLearnReductionTool(PCA(n_components=3)).random_state is None

        caller_state = np.random.RandomState(3)
        reducer = PCA(n_components=3, random_state=caller_state)
        rd = GoldSKLearnReductionTool(reducer)
        assert reducer.random_state is caller_state
        state = rd.random_state
        assert state is not None
        expected_state = np.random.RandomState(3).get_state(legacy=False)
        assert state["bit_generator"] == expected_state["bit_generator"]
        assert np.array_equal(state["state"]["key"], expected_state["state"]["key"])
        assert state["state"]["pos"] == expected_state["state"]["pos"]
        assert state["has_gauss"] == expected_state["has_gauss"]
        assert state["gauss"] == expected_state["gauss"]

        override = PCA(n_components=3, random_state=caller_state)
        GoldSKLearnReductionTool(override, random_state=5)
        assert isinstance(override.random_state, np.random.RandomState)
        assert override.random_state is not caller_state

    @pytest.mark.parametrize(
        "make_reducer",
        [
            lambda: PCA(n_components=3, svd_solver="randomized"),
            lambda: TSNE(n_components=2, perplexity=3, init="random"),
            lambda: UMAP(n_components=2, n_neighbors=5),
            lambda: GaussianRandomProjection(n_components=4),
        ],
        ids=["pca", "tsne", "umap", "gaussian_random_projection"],
    )
    def test_random_state_roundtrip_restores_output(self, make_reducer):
        data = torch.randn(20, 5, dtype=torch.float32)
        rd = GoldSKLearnReductionTool(make_reducer(), random_state=0)
        state = rd.random_state
        assert state is not None

        first = rd.fit_transform(data)
        advanced = rd.random_state
        assert advanced is not None
        assert not np.array_equal(advanced["state"]["key"], state["state"]["key"])

        rd.update_random_state(state)
        assert torch.allclose(rd.fit_transform(data), first)

        expected = GoldSKLearnReductionTool(
            make_reducer(), random_state=0
        ).fit_transform(data)
        assert torch.allclose(first, expected)


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
