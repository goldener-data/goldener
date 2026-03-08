from abc import abstractmethod, ABC
from typing_extensions import assert_never
from typing import Dict, List, Callable, Any

from dataclasses import dataclass
from enum import Enum

import torch


class GoldEmbeddingTool(ABC):
    """Abstract base class for embedding extraction from models.

    This class defines the interface for embedding tools that can embed and optionally
    fuse embeddings from models. Implementations should provide specific mechanisms for
    extracting embeddings from different types of models (e.g., PyTorch, multimodal).
    """

    @abstractmethod
    def embed(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Embed the input data using the model.

        Returns: Dictionary mapping layer names to their embedding tensors.
        """

    @abstractmethod
    def embed_and_fuse(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Embed and fuse the input data using the model.

        Returns: Fused embedding tensor.
        """


class EmbeddingFusionStrategy(Enum):
    """Strategies to fuse embeddings from multiple layers.

    CONCAT: Concatenate embeddings along the channel dimension.
    ADD: Element-wise addition of embeddings.
    AVERAGE: Element-wise average of embeddings.
    MAX: Element-wise maximum of embeddings.
    """

    CONCAT = "concat"
    ADD = "add"
    AVERAGE = "average"
    MAX = "max"


def fuse_tensors(
    tensors: List[torch.Tensor],
    strategy: EmbeddingFusionStrategy,
) -> torch.Tensor:
    """Apply the specified embedding fusion strategy to a list of tensors.

    Args:
        tensors: List of tensors to be fused.
        strategy: Strategy to fuse the tensors.

    Returns: Fused tensor.

    Raises:
        ValueError: If the tensors have a different number of dimensions.
    """
    ndims = set(f.ndim for f in tensors)
    if len(ndims) != 1:
        raise ValueError("All embeddings must have the same number of dimensions.")

    if strategy is EmbeddingFusionStrategy.CONCAT:
        return torch.cat(tensors, dim=1)
    elif strategy is EmbeddingFusionStrategy.ADD:
        return torch.stack(tensors, dim=0).sum(dim=0)
    elif strategy is EmbeddingFusionStrategy.AVERAGE:
        return torch.stack(tensors, dim=0).mean(dim=0)
    elif strategy is EmbeddingFusionStrategy.MAX:
        return torch.stack(tensors, dim=0).max(dim=0).values
    else:
        assert_never(strategy)


@dataclass
class GoldTorchEmbeddingToolConfig:
    """Configuration for the GoldTorchEmbeddingTool.

    Attributes:
        model: The PyTorch model from which to extract embeddings.
        layers: List of layer names or a dictionary mapping group names to lists of layer names.
            If None, the last layer of the model is used.
        layer_fusion: Strategy to fuse embeddings from multiple layers within the same group.
        group_fusion: Strategy to fuse embeddings from different groups.
    """

    model: torch.nn.Module
    layers: list[str] | dict[str, list[str]] | None = None
    layer_fusion: EmbeddingFusionStrategy = EmbeddingFusionStrategy.CONCAT
    group_fusion: EmbeddingFusionStrategy = EmbeddingFusionStrategy.CONCAT


class GoldEmbeddingFusionTool:
    """Embedding fusion from multiple layers and groups.

    Attributes:
        layer_fusion: Strategy to fuse embeddings from multiple layers within the same group.
        group_fusion: Strategy to fuse embeddings from different groups.
    """

    def __init__(
        self,
        layer_fusion: EmbeddingFusionStrategy = EmbeddingFusionStrategy.CONCAT,
        group_fusion: EmbeddingFusionStrategy = EmbeddingFusionStrategy.CONCAT,
    ) -> None:
        """Initialize the GoldEmbeddingFusionTool.

        Args:
            layer_fusion: Strategy to fuse embeddings from multiple layers within the same group.
                Defaults to CONCAT.
            group_fusion: Strategy to fuse embeddings from different groups. Defaults to CONCAT.
        """
        self.layer_fusion = layer_fusion
        self.group_fusion = group_fusion

    @staticmethod
    def fuse_tensors(
        tensors: List[torch.Tensor],
        strategy: EmbeddingFusionStrategy,
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
            raise ValueError("All embeddings must have the same number of dimensions.")

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
                        embedding,
                        size=max_size,
                        mode=mode,
                    )
                    if embedding.shape[2:] != max_size
                    else embedding
                )
                for embedding in tensors
            ]

        return fuse_tensors(tensors, strategy)

    def fuse_embeddings(
        self,
        x: dict[str, torch.Tensor],
        layers: list[str] | dict[str, list[str]],
    ) -> torch.Tensor:
        """Fuse embeddings from multiple layers and groups.

        Args:
            x: Dictionary mapping layer names to embedding tensors.
            layers: List of layer names or a dictionary mapping group names to lists of layer names.

        Returns: Fused embedding tensor.
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


class GoldTorchEmbeddingTool(GoldEmbeddingTool):
    """Embedding tool for PyTorch models.

    Once initialized, the tool registers forward hooks on the specified layers of the model.
    When the model processes input data, the hooks capture the outputs of these layers.
    The extracted embeddings can then be fused according to the specified strategies.

    The model and layers cannot be changed after initialization. The embedding fusion can be changed.

    Attributes:
        _model: The PyTorch model from which to extract embeddings.
        fusion: GoldEmbeddingFusionTool instance to handle embedding fusion.
        _layers: List of layer names or a dictionary mapping group names to lists of layer names.
        _hooks: Dictionary mapping layer names to their corresponding forward hook handles.
        _embeddings: Dictionary to store extracted embeddings.
    """

    def __init__(
        self,
        config: GoldTorchEmbeddingToolConfig,
    ) -> None:
        """Initialize the GoldTorchEmbeddingTool.

        Args:
            config: Configuration object containing the model, layers, and fusion strategies.
        """
        self._model = config.model
        self.fusion = GoldEmbeddingFusionTool(
            layer_fusion=config.layer_fusion,
            group_fusion=config.group_fusion,
        )

        self._layers: list[str] | dict[str, list[str]]
        self._hooks: dict[str, torch.utils.hooks.RemovableHandle]
        self._embeddings: dict[str, torch.Tensor]

        self._register_layers(config.layers)

    @property
    def model(self) -> torch.nn.Module:
        """The PyTorch model from which to extract embeddings."""
        return self._model

    @property
    def layers(self) -> list[str] | dict[str, list[str]]:
        """The layers from which to extract embeddings.

        It also indicates the grouping of layers for fusion.
        """
        return self._layers

    def embed(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Embed the input data using the model.

        Args:
            x: Input data tensor to be processed by the model.

        Returns: Dictionary mapping layer names to their embedding tensors.
        """
        self._embeddings = {}
        self._model(x)
        return self._embeddings

    def embed_and_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Embed and fuse the input data using the model.

        Args:
            x: Input data tensor to be processed by the model.

        Returns: Fused embedding tensor.
        """
        embeddings = self.embed(x)
        return self.fusion.fuse_embeddings(embeddings, self._layers)

    def __del__(self):
        """Remove all registered hooks when the embedder is deleted."""
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

        self._embeddings = {}

        def _get_hook(
            name: str,
        ) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
            def hook(
                module: torch.nn.Module,
                input: torch.Tensor,
                output: torch.Tensor,
            ) -> None:
                self._embeddings[name] = output.detach()

            return hook

        self._hooks = {
            name: module.register_forward_hook(_get_hook(name))
            for name, module in named_modules
            if name in layer_names
        }

        not_found = set(layer_names).difference(set(self._hooks.keys()))
        if not_found:
            raise ValueError(f"Layers not found in the model: {not_found}")


class GoldMultiModalTorchEmbeddingTool(GoldEmbeddingTool):
    """Embedding tool for multimodal data using PyTorch.

    Each modality has its own GoldTorchEmbeddingTool defined by its own configuration.
    This allows for processing different types of input data (e.g., images, text, audio)
    with different models and then fusing their embeddings.

    Attributes:
        embedders: Dictionary mapping modality names to their GoldTorchEmbeddingTool instances.
        strategy: Strategy for fusing embeddings from different modalities.
    """

    def __init__(
        self,
        configs: Dict[str, GoldTorchEmbeddingToolConfig],
        strategy: EmbeddingFusionStrategy = EmbeddingFusionStrategy.CONCAT,
    ) -> None:
        """Initialize the multimodal embedding tool.

        Args:
            configs: Dictionary mapping modality names to their GoldTorchEmbeddingToolConfig.
            strategy: Strategy to use for fusing embeddings from different modalities. Defaults to CONCAT.
        """
        self.embedders = {
            modality: GoldTorchEmbeddingTool(config)
            for modality, config in configs.items()
        }
        self.strategy = strategy

    def embed_and_fuse(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed and fuse multimodal input data.

        Args:
            x: Dictionary mapping modality names to their input tensors.

        Returns:
            Fused embedding tensor combining all modalities.
        """
        return GoldEmbeddingFusionTool.fuse_tensors(
            [
                tool.embed_and_fuse(x[modality])
                for modality, tool in self.embedders.items()
            ],
            self.strategy,
        )

    def embed(self, x: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Embed multimodal input data without fusing.

        Args:
            x: Dictionary mapping modality names to their input tensors.

        Returns:
            Dictionary mapping "{modality}.{layer}" to their embedding tensors.
        """
        per_modality = {
            modality: tool.embed(x[modality])
            for modality, tool in self.embedders.items()
        }

        return {
            f"{modality}.{layer}": embedding
            for modality, embeddings in per_modality.items()
            for layer, embedding in embeddings.items()
        }
