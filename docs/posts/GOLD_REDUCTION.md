# Reduce embedding width and keep a common reference space

Wide embeddings can make repeated similarity searches expensive. Reducing the feature width can save memory and speed up later searches. For example, a job that monitors sensor behavior can compare this week's signal windows with a reference archive in a smaller feature space. Fit the projection once so changes in the coordinates reflect changes in the inputs under that mapping.

`GoldSKLearnReductionTool` wraps PCA and other estimators for PyTorch tensors. Fit PCA once on representative reference vectors, then apply that fitted projection to later windows. For training and validation, learn the mapping on training vectors and reuse it on validation vectors.

## Fit on a reference and project a later window

The toy tensors stand in for precomputed embeddings. PCA compresses four features to two and preserves each window's row count. One later row duplicates the first reference row to show that both calls use the same fitted coordinates.

```python
import torch
from sklearn.decomposition import PCA

from goldener import GoldSKLearnReductionTool

reference = torch.tensor(
    [
        [0.0, 0.0, 0.1, 0.0],
        [0.1, 0.2, 0.0, 0.1],
        [0.0, 0.3, 0.2, 0.1],
        [4.0, 4.0, 0.0, 0.3],
        [4.2, 3.9, 0.1, 0.2],
        [4.1, 4.2, 0.0, 0.0],
    ],
)
later = torch.tensor([[0.0, 0.0, 0.1, 0.0], [4.3, 4.0, 0.2, 0.1]])
pca = PCA(n_components=2)
reducer = GoldSKLearnReductionTool(pca)

reference_reduced = reducer.fit_transform(reference)
later_reduced = reducer.transform(later)

assert torch.allclose(reference_reduced[0], later_reduced[0], atol=1e-6)
print(reference_reduced.shape, later_reduced.shape)  # (6, 2), (2, 2)
```

Reduction expects finite floating tensors of shape `(num_vectors, feature_dim)`. Row order carries sample and vector identity through the projection. Calling `fit` or `fit_transform` again replaces the fitted mapping. The adapter runs the estimator on detached CPU NumPy arrays and returns a tensor on the input device with the input dtype. Store the fitted estimator (`pca` here) alongside the embedding and preprocessing configuration when later jobs need to use the same space.

## Choose a projection that fits the job

| Method | What the projection does | What to expect for later rows |
| --- | --- | --- |
| PCA | Keeps directions of high variance in the reference data. | Applies the same fitted linear mapping. [PCA review](https://doi.org/10.1098/rsta.2015.0202) |
| Gaussian random projection | Draws a linear mapping that can approximately preserve pairwise distances with enough output dimensions. | Applies the same sampled matrix. [Random projection and the Johnson–Lindenstrauss lemma](https://doi.org/10.1002/rsa.10073) |
| UMAP | Learns a nonlinear embedding that emphasizes local neighborhoods. | Transforms points into the fitted embedding. [UMAP paper](https://arxiv.org/abs/1802.03426) |
| t-SNE | Builds a visualization of the rows in one fit. | Goldener warns and refits if you call `transform`, creating a new map. Use `fit_transform` for a one-off plot. [t-SNE paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html) |

`GoldTorchModuleReductionTool` provides the same `transform` entry point for a PyTorch module you already have, such as a trained projection head. The module handles its own device and dtype. Supply a module that returns one row per input row in the same order. The wrapper applies the module as provided.

## Check what the smaller vectors preserve

A projection can discard cues that matter for your downstream task. If you reduce image embeddings before searching for near-duplicates, verify that crops or resizes of the same image still find one another and that visually similar images of different subjects stay distinguishable. PCA can drop a direction that carries little total variance yet separates the cases you care about. Choose an output width by measuring resource savings and checking the effect on those cases.

Goldener's clusterizer, selector, and density weigher can apply a reducer inside their workflow. A fittable reducer is fitted on the data being processed, which can mean a separate fit per label or chunk. To compare later samples in a shared space, compute the projection once and pass the consistently projected vectors into those stages. [Clustering embeddings](GOLD_CLUSTERIZER.md) · [Weighting samples by embedding density](GOLD_DENSITY_WEIGHING.md)
