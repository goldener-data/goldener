from abc import abstractmethod
from typing_extensions import assert_never
from typing import Dict, List, Callable, Any

from dataclasses import dataclass
from enum import Enum

import torch


class GoldFeatureExtractor:
    @abstractmethod
    def extract(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Extract features from the model for the given input data.

        Returns: Dictionary mapping layer names to their extracted feature tensors.
        """

    @abstractmethod
    def extract_and_fuse(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Extract and fuse features from the model for the given input data.

        Returns: Fused feature tensor.
        """


class FeatureFusionStrategy(Enum):
    """Strategies to fuse features from multiple layers.

    CONCAT: Concatenate features along the channel dimension.
    ADD: Element-wise addition of features.
    AVERAGE: Element-wise average of features.
    MAX: Element-wise maximum of features.
    NO_FUSION: No fusion, return features as is (only valid for single layer).
    """

    CONCAT = "concat"
    ADD = "add"
    AVERAGE = "average"
    MAX = "max"


@dataclass
class TorchGoldFeatureExtractorConfig:
    """Configuration for the TorchFeatureExtractor.

    Attributes:
        model: The PyTorch model from which to extract features.
        layers: List of layer names or a dictionary mapping group names to lists of layer names.
            If None, the last layer of the model is used.
        layer_fusion: Strategy to fuse features from multiple layers within the same group.
        group_fusion: Strategy to fuse features from different groups.
    """

    model: torch.nn.Module
    layers: list[str] | dict[str, list[str]] | None = None
    layer_fusion: FeatureFusionStrategy = FeatureFusionStrategy.CONCAT
    group_fusion: FeatureFusionStrategy = FeatureFusionStrategy.CONCAT


class GoldFeatureFusion:
    """Feature fusion from multiple layers and groups.

    Attributes:
        layer_fusion: Strategy to fuse features from multiple layers within the same group.
        group_fusion: Strategy to fuse features from different groups.
    """

    def __init__(
        self,
        layer_fusion: FeatureFusionStrategy = FeatureFusionStrategy.CONCAT,
        group_fusion: FeatureFusionStrategy = FeatureFusionStrategy.CONCAT,
    ) -> None:
        self.layer_fusion = layer_fusion
        self.group_fusion = group_fusion

    @staticmethod
    def fuse_tensors(
        tensors: List[torch.Tensor],
        strategy: FeatureFusionStrategy,
    ) -> torch.Tensor:
        """Fuse a list of tensors.

        The tensors are at least expected to have the shape (B, C), until (B, C, D, H, W) if sizes after the
        channel dimension differ. In this case, all tensors are interpolated to the largest size.

        Args:
            tensors: List of tensors to be fused.
            strategy: Strategy to fuse the tensors.

        Returns: Fused tensors.

        Raises:
            ValueError: If the tensors have a different number of dimensions.
        """
        ndims = set(f.ndim for f in tensors)
        if len(ndims) != 1:
            raise ValueError("All features must have the same number of dimensions.")

        ndim = list(ndims)[0]
        if ndim > 2:
            max_size = tuple(
                max(sizes) for sizes in zip(*(f.shape[2:] for f in tensors))
            )
            # Interpolate all tensors to the largest size
            mode = "linear" if ndim == 3 else ("bilinear" if ndim == 4 else "trilinear")
            tensors = [
                (
                    torch.nn.functional.interpolate(
                        feature,
                        size=max_size,
                        mode=mode,
                    )
                    if feature.shape[2:] != max_size
                    else feature
                )
                for feature in tensors
            ]

        if strategy is FeatureFusionStrategy.CONCAT:
            return torch.cat(tensors, dim=1)
        elif strategy is FeatureFusionStrategy.ADD:
            return torch.stack(tensors, dim=0).sum(dim=0)
        elif strategy is FeatureFusionStrategy.AVERAGE:
            return torch.stack(tensors, dim=0).mean(dim=0)
        elif strategy is FeatureFusionStrategy.MAX:
            return torch.stack(tensors, dim=0).max(dim=0).values
        else:
            assert_never(strategy)

    def fuse_features(
        self,
        x: dict[str, torch.Tensor],
        layers: list[str] | dict[str, list[str]],
    ) -> torch.Tensor:
        """Fuse features from multiple layers and groups.

        Args:
            x: Dictionary mapping layer names to feature tensors.
            layers: List of layer names or a dictionary mapping group names to lists of layer names.

        Returns: Fused feature tensor.
        """
        # list of layers are fused by layer_fusion strategy
        if isinstance(layers, list):
            return self.fuse_tensors([x[name] for name in layers], self.layer_fusion)

        # groups of layers are fused by layer_fusion strategy,
        # then all groups are fused by group_fusion strategy
        fused_groups = []
        for group, layer_names in layers.items():
            if len(layer_names) == 1:
                fused_groups.append(x[layer_names[0]])
            else:
                fused_groups.append(
                    self.fuse_tensors(
                        [x[name] for name in layer_names], self.layer_fusion
                    )
                )
        return self.fuse_tensors(fused_groups, self.group_fusion)


class TorchGoldFeatureExtractor(GoldFeatureExtractor):
    """Feature extractor for PyTorch models.

    Once initialized, the extractor registers forward hooks on the specified layers of the model.
    When the model processes input data, the hooks capture the outputs of these layers.
    The extracted features can then be fused according to the specified strategies.

    The model and layers cannot be changed after initialization. The feature fusion can be changed.

    Attributes:
        _model: The PyTorch model from which to extract features.
        fusion: FeatureFusion instance to handle feature fusion.
        _layers: List of layer names or a dictionary mapping group names to lists of layer names.
        _hooks: Dictionary mapping layer names to their corresponding forward hook handles.
        _features: Dictionary to store extracted features.
    """

    def __init__(
        self,
        config: TorchGoldFeatureExtractorConfig,
    ) -> None:
        self._model = config.model
        self.fusion = GoldFeatureFusion(
            layer_fusion=config.layer_fusion,
            group_fusion=config.group_fusion,
        )

        self._layers: list[str] | dict[str, list[str]]
        self._hooks: dict[str, torch.utils.hooks.RemovableHandle]
        self._features: dict[str, torch.Tensor]

        self._register_layers(config.layers)

    @property
    def model(self) -> torch.nn.Module:
        """The PyTorch model from which to extract features."""
        return self._model

    @property
    def layers(self) -> list[str] | dict[str, list[str]]:
        """The layers from which to extract features.

        It also indicates the grouping of layers for fusion.
        """
        return self._layers

    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from the model for the given input data.

        Args:
            x: Input data tensor to be processed by the model.

        Returns: Dictionary mapping layer names to their extracted feature tensors.
        """
        self._features = {}
        self._model(x)
        return self._features

    def extract_and_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and fuse features from the model for the given input data.

        Args:
            x: Input data tensor to be processed by the model.

        Returns: Fused feature tensor.
        """
        features = self.extract(x)
        return self.fusion.fuse_features(features, self._layers)

    def __del__(self):
        """Remove all registered hooks when the extractor is deleted."""
        for handle in self._hooks.values():
            handle.remove()

    def _register_layers(self, layers: List[str] | Dict[str, List[str]] | None) -> None:
        """Register forward hooks on the specified layers of the model.

        Args:
            layers: List of layer names or a dictionary mapping group names to lists of layer names.
                If None, the last layer of the model is used.
        """

        named_modules = list(self._model.named_modules())
        if layers is None:
            layer_names = [named_modules[-1][0]]  # last layer
            layers = layer_names
        elif isinstance(layers, dict):
            layer_names = [name for names in layers.values() for name in names]
        elif isinstance(layers, list):
            layer_names = layers
        else:
            assert_never(layers)

        self._layers = layers

        self._features = {}

        def _get_hook(
            name: str,
        ) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
            def hook(
                module: torch.nn.Module,
                input: torch.Tensor,
                output: torch.Tensor,
            ) -> None:
                self._features[name] = output.detach()

            return hook

        self._hooks = {
            name: module.register_forward_hook(_get_hook(name))
            for name, module in named_modules
            if name in layer_names
        }

        not_found = set(layer_names).difference(set(self._hooks.keys()))
        if not_found:
            raise ValueError(f"Layers not found in the model: {not_found}")


class MultiModalTorchGoldFeatureExtractor(GoldFeatureExtractor):
    """Feature extractor for multimodal data using PyTorch.

    Each modality has its own TorchFeatureExtractor defined by its own configuration.
    """

    def __init__(
        self,
        configs: Dict[str, TorchGoldFeatureExtractorConfig],
        strategy: FeatureFusionStrategy = FeatureFusionStrategy.CONCAT,
    ) -> None:
        self.extractors = {
            modality: TorchGoldFeatureExtractor(config)
            for modality, config in configs.items()
        }
        self.strategy = strategy

    def extract_and_fuse(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return GoldFeatureFusion.fuse_tensors(
            [
                extractor.extract_and_fuse(x[modality])
                for modality, extractor in self.extractors.items()
            ],
            self.strategy,
        )

    def extract(self, x: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        per_modality = {
            modality: extractor.extract(x[modality])
            for modality, extractor in self.extractors.items()
        }

        return {
            f"{modality}.{layer}": feature
            for modality, features in per_modality.items()
            for layer, feature in features.items()
        }
