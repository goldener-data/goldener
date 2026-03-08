import pytest

import torch

from goldener.embed import (
    GoldEmbeddingFusionTool,
    EmbeddingFusionStrategy,
    GoldTorchEmbeddingTool,
    GoldTorchEmbeddingToolConfig,
    GoldMultiModalTorchEmbeddingTool,
    fuse_tensors,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


def make_tensor(shape, fill_value=None):
    if fill_value is not None:
        return torch.full(shape, fill_value)
    return torch.randn(shape)


shapes_2d_3d_4d = [
    (2, 4),
    (2, 4, 5),
    (2, 4, 5, 5),
    (2, 4, 5, 5, 5),
]


class TestFuseTensors:
    def test_top_level_basic_concat(self) -> None:
        t1 = torch.ones(2, 3)
        t2 = torch.full((2, 3), 2.0)

        fused = fuse_tensors([t1, t2], EmbeddingFusionStrategy.CONCAT)

        assert fused.shape == (2, 6)
        assert torch.allclose(fused[:, :3], torch.ones_like(fused[:, :3]))
        assert torch.allclose(fused[:, 3:], torch.full_like(fused[:, 3:], 2.0))

    def test_top_level_add(self) -> None:
        t1 = torch.full((2, 3), 1.0)
        t2 = torch.full((2, 3), 3.0)

        fused = fuse_tensors([t1, t2], EmbeddingFusionStrategy.ADD)

        assert fused.shape == (2, 3)
        assert torch.allclose(fused, torch.full_like(fused, 4.0))

    def test_top_level_average(self) -> None:
        t1 = torch.full((2, 3), 1.0)
        t2 = torch.full((2, 3), 3.0)

        fused = fuse_tensors([t1, t2], EmbeddingFusionStrategy.AVERAGE)

        assert fused.shape == (2, 3)
        assert torch.allclose(fused, torch.full_like(fused, 2.0))

    def test_top_level_max(self) -> None:
        t1 = torch.tensor([[1.0, 5.0, 2.0], [0.0, 3.0, 4.0]])
        t2 = torch.tensor([[2.0, 4.0, 1.0], [1.0, 1.0, 5.0]])

        fused = fuse_tensors([t1, t2], EmbeddingFusionStrategy.MAX)

        expected = torch.max(t1, t2)
        assert fused.shape == t1.shape
        assert torch.allclose(fused, expected)

    def test_top_level_raises_on_mismatched_ndim(self) -> None:
        t1 = torch.ones(2, 3)
        t2 = torch.ones(2, 3, 4)

        with pytest.raises(ValueError, match="same number of dimensions"):
            fuse_tensors([t1, t2], EmbeddingFusionStrategy.ADD)


class TestGoldEmbeddingFusionTool:
    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_tensors_concat(self, shape):
        t1 = make_tensor(shape)
        t2 = make_tensor(shape)
        fused = GoldEmbeddingFusionTool.fuse_tensors(
            [t1, t2], EmbeddingFusionStrategy.CONCAT
        )
        assert fused.shape[1] == shape[1] * 2
        assert fused.shape[:2] == (shape[0], shape[1] * 2)[:2]

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_tensors_add(self, shape):
        t1 = make_tensor(shape, 1.0)
        t2 = make_tensor(shape, 1.0)
        fused = GoldEmbeddingFusionTool.fuse_tensors(
            [t1, t2], EmbeddingFusionStrategy.ADD
        )
        assert torch.allclose(fused, torch.full_like(fused, 2.0))

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_tensors_average(self, shape):
        t1 = make_tensor(shape, 1.0)
        t2 = make_tensor(shape, 3.0)
        fused = GoldEmbeddingFusionTool.fuse_tensors(
            [t1, t2], EmbeddingFusionStrategy.AVERAGE
        )
        assert torch.allclose(fused, torch.full_like(fused, 2.0))

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_tensors_max(self, shape):
        t1 = make_tensor(shape, 1.0)
        t2 = make_tensor(shape, 3.0)

        fused = GoldEmbeddingFusionTool.fuse_tensors(
            [t1, t2], EmbeddingFusionStrategy.MAX
        )

        assert fused.shape == t1.shape
        assert torch.allclose(fused, torch.full_like(fused, 3.0))

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_tensors_with_different_shapes(self, shape):
        # Only test for 3D and 4D shapes (spatial dims)
        if len(shape) < 3:
            return
        t1 = make_tensor(shape)
        smaller_shape = (shape[0], shape[1]) + tuple(max(1, s // 2) for s in shape[2:])
        t2 = make_tensor(smaller_shape)
        fused = GoldEmbeddingFusionTool.fuse_tensors(
            [t1, t2], EmbeddingFusionStrategy.ADD
        )
        assert fused.shape == shape

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_embeddings(self, shape):
        t1 = make_tensor(shape)
        t2 = make_tensor(shape)
        embeddings = {"mod1": t1, "mod2": t2}
        fusion = GoldEmbeddingFusionTool(
            layer_fusion=EmbeddingFusionStrategy.ADD,
            group_fusion=EmbeddingFusionStrategy.ADD,
        )
        fused = fusion.fuse_embeddings(embeddings, ["mod1", "mod2"])
        assert fused.shape == shape

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_embeddings_with_groups(self, shape):
        t1 = make_tensor(shape)
        t2 = make_tensor(shape)
        t3 = make_tensor(shape)
        embeddings = {"layer1": t1, "layer2": t2, "layer3": t3}
        fusion = GoldEmbeddingFusionTool(
            layer_fusion=EmbeddingFusionStrategy.ADD,
            group_fusion=EmbeddingFusionStrategy.CONCAT,
        )
        fused = fusion.fuse_embeddings(
            embeddings, {"m1": ["layer1", "layer2"], "m2": ["layer3"]}
        )
        # Should concatenate along channel dim
        assert fused.shape[1] == shape[1] * 2
        assert fused.shape[0] == shape[0]
        if len(shape) > 2:
            assert fused.shape[2:] == shape[2:]

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_embeddings_with_max_layer_fusion(self, shape):
        t1 = make_tensor(shape, 1.0)
        t2 = make_tensor(shape, 3.0)
        embeddings = {"mod1": t1, "mod2": t2}
        fusion = GoldEmbeddingFusionTool(
            layer_fusion=EmbeddingFusionStrategy.MAX,
            group_fusion=EmbeddingFusionStrategy.ADD,
        )
        fused = fusion.fuse_embeddings(embeddings, ["mod1", "mod2"])

        assert fused.shape == shape
        assert torch.allclose(fused, torch.full_like(fused, 3.0))


class TestGoldTorchEmbeddingTool:
    def test_embed(self):
        model = DummyModel()
        layers = ["conv1", "conv2"]
        config = GoldTorchEmbeddingToolConfig(model=model, layers=layers)
        tool = GoldTorchEmbeddingTool(config)
        data = torch.randn(2, 3, 8, 8)
        embeddings = tool.embed(data)
        assert len(embeddings) == len(layers)
        assert embeddings["conv1"].shape == (2, 4, 8, 8)
        assert embeddings["conv2"].shape == (2, 8, 8, 8)

    def test_embed_and_fuse(self):
        model = DummyModel()
        layers = ["conv1", "conv2"]
        config = GoldTorchEmbeddingToolConfig(
            model=model,
            layers=layers,
            layer_fusion=EmbeddingFusionStrategy.CONCAT,
        )
        tool = GoldTorchEmbeddingTool(config)
        data = torch.randn(2, 3, 8, 8)
        fused = tool.embed_and_fuse(data)
        # Should add embeddings from conv1 and conv2
        assert fused.shape == (2, 12, 8, 8)

    def test_invalid_layer(self):
        model = DummyModel()
        config = GoldTorchEmbeddingToolConfig(model=model, layers=["invalid_layer"])
        with pytest.raises(ValueError):
            GoldTorchEmbeddingTool(config)


class TestGoldMultiModalTorchEmbeddingTool:
    def test_embed(self):
        model1 = DummyModel()
        model2 = DummyModel()
        config1 = GoldTorchEmbeddingToolConfig(model=model1, layers=["conv1"])
        config2 = GoldTorchEmbeddingToolConfig(model=model2, layers=["conv2"])
        tool = GoldMultiModalTorchEmbeddingTool({"img": config1, "aux": config2})
        data = {
            "img": torch.randn(2, 3, 8, 8),
            "aux": torch.randn(2, 3, 8, 8),
        }
        embeddings = tool.embed(data)
        assert len(embeddings) == 2
        assert embeddings["img.conv1"].shape == (2, 4, 8, 8)
        assert embeddings["aux.conv2"].shape == (2, 8, 8, 8)

    def test_embed_and_fuse(self):
        model1 = DummyModel()
        model2 = DummyModel()
        config1 = GoldTorchEmbeddingToolConfig(model=model1, layers=["conv1"])
        config2 = GoldTorchEmbeddingToolConfig(model=model2, layers=["conv2"])
        tool = GoldMultiModalTorchEmbeddingTool(
            {"img": config1, "aux": config2},
            strategy=EmbeddingFusionStrategy.CONCAT,
        )
        data = {
            "img": torch.randn(2, 3, 8, 8),
            "aux": torch.randn(2, 3, 8, 8),
        }
        fused = tool.embed_and_fuse(data)
        # Should concatenate embeddings from both modalities
        assert fused.shape[0] == 2
        assert fused.shape[2:] == (8, 8)
