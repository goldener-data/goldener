"""Goldener - Make your data even more valuable.

Goldener is a modality-agnostic library to orchestrate data during the full lifecycle of AI pipelines. All its
features rely on the same principle: the semantics of the data is described by embeddings extracted from
pre-trained/foundational models, and data-centric algorithms are applied on these embeddings.

The features are built as a pipeline of steps, each step storing its results locally so that it can be stopped
and restarted without recomputing what is already done:

- `goldener.describe`: compute the embeddings of the samples of a dataset.
- `goldener.vectorize`: turn the embeddings into 2D vectors (one or multiple vectors per sample).
- `goldener.reduce`: reduce the dimension of the vectors.
- `goldener.clusterize`: group the vectors in clusters.
- `goldener.select`: select the most representative subset of samples.
- `goldener.split`: split a dataset in multiple sets (train, validation, test, ...).
- `goldener.organize`: balance the batches with the clusters of the data during training.

The main classes of every file are re-exported at the root of the package, e.g. `from goldener import GoldSplitter`.
"""

from goldener.clusterize import (
    GoldClusteringTool,
    GoldSKLearnClusteringTool,
    GoldRandomClusteringTool,
    GoldClusterizer,
)
from goldener.describe import (
    GoldDescriptor,
)
from goldener.embed import (
    EmbeddingFusionStrategy,
    GoldEmbeddingFusionTool,
    GoldEmbeddingTool,
    GoldTorchEmbeddingTool,
    GoldTorchEmbeddingToolConfig,
    GoldMultiModalTorchEmbeddingTool,
)
from goldener.organize import GoldClusterizedBatchSampler
from goldener.pxt_utils import GoldPxtTorchDataset
from goldener.reduce import (
    GoldReductionTool,
    GoldReductionToolWithFit,
    GoldSKLearnReductionTool,
    GoldTorchModuleReductionTool,
)
from goldener.select import (
    GoldSelectionTool,
    GoldSelector,
    GoldGreedyClosestPointSelectionTool,
    GoldGreedyFarthestPointSelectionTool,
    GoldGreedyKCenterSelectionTool,
    GoldGreedyKernelPointsSelectionTool,
)
from goldener.split import GoldSet, GoldSplitter
from goldener.torch_utils import ResetableTorchIterableDataset
from goldener.vectorize import (
    Filter2DWithCount,
    FilterLocation,
    Vectorized,
    GoldTensorVectorizationTool,
)


__all__ = (
    "GoldClusteringTool",
    "GoldSKLearnClusteringTool",
    "GoldRandomClusteringTool",
    "GoldClusterizer",
    "GoldDescriptor",
    "EmbeddingFusionStrategy",
    "GoldEmbeddingFusionTool",
    "GoldEmbeddingTool",
    "GoldTorchEmbeddingTool",
    "GoldTorchEmbeddingToolConfig",
    "GoldMultiModalTorchEmbeddingTool",
    "GoldClusterizedBatchSampler",
    "GoldPxtTorchDataset",
    "GoldReductionTool",
    "GoldReductionToolWithFit",
    "GoldSKLearnReductionTool",
    "GoldTorchModuleReductionTool",
    "GoldSelectionTool",
    "GoldSelector",
    "GoldGreedyClosestPointSelectionTool",
    "GoldGreedyFarthestPointSelectionTool",
    "GoldGreedyKCenterSelectionTool",
    "GoldGreedyKernelPointsSelectionTool",
    "GoldSet",
    "GoldSplitter",
    "ResetableTorchIterableDataset",
    "Filter2DWithCount",
    "FilterLocation",
    "Vectorized",
    "GoldTensorVectorizationTool",
)
