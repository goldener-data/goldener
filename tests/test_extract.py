import pytest

import torch

from goldener.extract import (
    GoldFeatureFusion,
    FeatureFusionStrategy,
    TorchGoldFeatureExtractor,
    TorchGoldFeatureExtractorConfig,
    MultiModalTorchGoldFeatureExtractor,
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


class TestFeatureFusion:
    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_concat(self, shape):
        t1 = make_tensor(shape)
        t2 = make_tensor(shape)
        fused = GoldFeatureFusion.fuse_tensors([t1, t2], FeatureFusionStrategy.CONCAT)
        assert fused.shape[1] == shape[1] * 2
        assert fused.shape[:2] == (shape[0], shape[1] * 2)[:2]

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_add(self, shape):
        t1 = make_tensor(shape, 1.0)
        t2 = make_tensor(shape, 1.0)
        fused = GoldFeatureFusion.fuse_tensors([t1, t2], FeatureFusionStrategy.ADD)
        assert torch.allclose(fused, torch.full_like(fused, 2.0))

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_average(self, shape):
        t1 = make_tensor(shape, 1.0)
        t2 = make_tensor(shape, 3.0)
        fused = GoldFeatureFusion.fuse_tensors([t1, t2], FeatureFusionStrategy.AVERAGE)
        assert torch.allclose(fused, torch.full_like(fused, 2.0))

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_with_different_shapes(self, shape):
        # Only test for 3D and 4D shapes (spatial dims)
        if len(shape) < 3:
            return
        t1 = make_tensor(shape)
        smaller_shape = (shape[0], shape[1]) + tuple(max(1, s // 2) for s in shape[2:])
        t2 = make_tensor(smaller_shape)
        fused = GoldFeatureFusion.fuse_tensors([t1, t2], FeatureFusionStrategy.ADD)
        assert fused.shape == shape

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_features(self, shape):
        t1 = make_tensor(shape)
        t2 = make_tensor(shape)
        features = {"mod1": t1, "mod2": t2}
        fusion = GoldFeatureFusion(
            layer_fusion=FeatureFusionStrategy.ADD,
            group_fusion=FeatureFusionStrategy.ADD,
        )
        fused = fusion.fuse_features(features, ["mod1", "mod2"])
        assert fused.shape == shape

    @pytest.mark.parametrize("shape", shapes_2d_3d_4d)
    def test_fuse_features_with_groups(self, shape):
        t1 = make_tensor(shape)
        t2 = make_tensor(shape)
        t3 = make_tensor(shape)
        features = {"layer1": t1, "layer2": t2, "layer3": t3}
        fusion = GoldFeatureFusion(
            layer_fusion=FeatureFusionStrategy.ADD,
            group_fusion=FeatureFusionStrategy.CONCAT,
        )
        fused = fusion.fuse_features(
            features, {"m1": ["layer1", "layer2"], "m2": ["layer3"]}
        )
        # Should concatenate along channel dim
        assert fused.shape[1] == shape[1] * 2
        assert fused.shape[0] == shape[0]
        if len(shape) > 2:
            assert fused.shape[2:] == shape[2:]


class TestTorchFeatureExtractor:
    def test_extract(self):
        model = DummyModel()
        layers = ["conv1", "conv2"]
        config = TorchGoldFeatureExtractorConfig(model=model, layers=layers)
        extractor = TorchGoldFeatureExtractor(config)
        data = torch.randn(2, 3, 8, 8)
        features = extractor.extract(data)
        assert len(features) == len(layers)
        assert features["conv1"].shape == (2, 4, 8, 8)
        assert features["conv2"].shape == (2, 8, 8, 8)

    def test_extract_and_fuse(self):
        model = DummyModel()
        layers = ["conv1", "conv2"]
        config = TorchGoldFeatureExtractorConfig(
            model=model,
            layers=layers,
            layer_fusion=FeatureFusionStrategy.CONCAT,
        )
        extractor = TorchGoldFeatureExtractor(config)
        data = torch.randn(2, 3, 8, 8)
        fused = extractor.extract_and_fuse(data)
        # Should add features from conv1 and conv2
        assert fused.shape == (2, 12, 8, 8)

    def test_invalid_layer(self):
        model = DummyModel()
        config = TorchGoldFeatureExtractorConfig(model=model, layers=["invalid_layer"])
        with pytest.raises(ValueError):
            TorchGoldFeatureExtractor(config)


class TestMultiModalTorchFeatureExtractor:
    def test_extract(self):
        model1 = DummyModel()
        model2 = DummyModel()
        config1 = TorchGoldFeatureExtractorConfig(model=model1, layers=["conv1"])
        config2 = TorchGoldFeatureExtractorConfig(model=model2, layers=["conv2"])
        extractor = MultiModalTorchGoldFeatureExtractor(
            {"img": config1, "aux": config2}
        )
        data = {
            "img": torch.randn(2, 3, 8, 8),
            "aux": torch.randn(2, 3, 8, 8),
        }
        features = extractor.extract(data)
        assert len(features) == 2
        assert features["img.conv1"].shape == (2, 4, 8, 8)
        assert features["aux.conv2"].shape == (2, 8, 8, 8)

    def test_extract_and_fuse(self):
        model1 = DummyModel()
        model2 = DummyModel()
        config1 = TorchGoldFeatureExtractorConfig(model=model1, layers=["conv1"])
        config2 = TorchGoldFeatureExtractorConfig(model=model2, layers=["conv2"])
        extractor = MultiModalTorchGoldFeatureExtractor(
            {"img": config1, "aux": config2},
            strategy=FeatureFusionStrategy.CONCAT,
        )
        data = {
            "img": torch.randn(2, 3, 8, 8),
            "aux": torch.randn(2, 3, 8, 8),
        }
        fused = extractor.extract_and_fuse(data)
        # Should concatenate features from both modalities
        assert fused.shape[0] == 2
        assert fused.shape[2:] == (8, 8)
