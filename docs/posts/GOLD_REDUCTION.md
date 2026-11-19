# GOLDENER <br> Reducing the dimensionality of embeddings

[**Introduction**](#1-introduction) |
[**Components**](#2-components) |
[**Selection and clustering**](#3-selection-and-clustering) |
[**Bibliography**](#4-bibliography)

## TLDR

High-dimensional embeddings make distance-based selection and clustering expensive in memory and time. [Goldener](https://github.com/goldener-data/goldener) can project those vectors with `GoldSKLearnReductionTool` or `GoldTorchModuleReductionTool` before the downstream computation. A fitted linear method such as PCA can then place new rows in the same mapping. The projection drops information, and a visualization method such as t-SNE does not provide that reusable mapping.

The following example runs with Goldener installed. The tensor stands in for embeddings that are already vectors. `fit_transform` learns the mapping from these rows and returns one reduced row per input row. `transform` applies that fitted PCA to later rows.

```python
import torch
from sklearn.decomposition import PCA

from goldener import GoldSKLearnReductionTool

# 50 embeddings of dimension 16.
x = torch.randn(50, 16, dtype=torch.float32)
reducer = GoldSKLearnReductionTool(PCA(n_components=2))
fitted = reducer.fit_transform(x)
print(fitted.shape, fitted.dtype, fitted.device)  # (50, 2), float32, cpu

new_rows = torch.randn(8, 16, dtype=torch.float32)
projected = reducer.transform(new_rows)
print(projected.shape, projected.dtype, projected.device)  # (8, 2), float32, cpu

# Row order matches the input. Changing x[0] changes reduced row 0 only.
same_rows = reducer.transform(x)
changed = x.clone()
changed[0] = changed[0] + 1
changed_rows = reducer.transform(changed)
print(torch.allclose(changed_rows[1], same_rows[1]))  # unchanged row
print(torch.allclose(changed_rows[0], same_rows[0]))  # changed row
```

This toy tensor shows the public tensor API. It is not a measured speedup. The scikit-learn adapter detaches the input, moves it to CPU for NumPy, and returns the result on the input device and dtype.

## 1. Introduction

Selection, clustering, and neighborhood scoring compare vectors. Their temporary matrices and distance work grow with the feature width. Embeddings from a foundation model are often hundreds or thousands of dimensions wide, so a smaller representation can make a downstream pass fit in memory and finish sooner. The reducer itself has a cost. The saving appears when that cost is smaller than the downstream work it replaces.

A shorter vector is a lossy view of the original embedding. Directions that PCA treats as low variance, or neighbors that a nonlinear method fails to preserve, can be exactly the differences a later model or review step needed. Check a sample of original neighbors before treating a reduced run as equivalent to the full embedding.

Two uses of a projection should stay separate:

- **Downstream computation.** PCA and Gaussian random projection learn a linear map and can place new vectors in that map with `transform`. UMAP also implements `transform`, with its own out-of-sample embedding. Use these when a later step must see both the fitted rows and new rows in one coordinate system.
- **Visualization of one fitted set.** t-SNE embeds the rows it is fit on. The scikit-learn estimator has no `transform`. It is a view of that set, useful for inspection, and it is not a map to apply to a later batch.

`GoldSelector` and `GoldClusterizer` can take one of these tools and reduce each population they fit. That integration is described below. A persisted `GoldReducer` that stores a fitted mapping for a later process is separate work; these tools keep their fitted estimator in memory.

## 2. Components

### 2.1. Tensor interfaces

Reduction operates on vectors that are already 2D. Every tool expects a finite floating tensor shaped `(num_vectors, feature_dim)` with a positive feature width. Row `i` is one vector. The four public types are exported from `goldener`:

- `GoldReductionTool` is the abstract interface. Subclasses implement `transform(x)`.
- `GoldReductionToolWithFit` extends that interface with `fit(x)` and `fit_transform(x)` for methods that learn from data.
- `GoldSKLearnReductionTool` is the fittable adapter for UMAP, PCA, t-SNE, and Gaussian random projection.
- `GoldTorchModuleReductionTool` wraps a `torch.nn.Module`. It only implements `transform`; it does not fit the module.

`GoldSKLearnReductionTool` accepts only those four estimator types. Passing one of them does not mean the four algorithms share a transform, a random-state contract, or a comparable geometry. Invalid inputs raise `ValueError` and name the concrete tool that rejected them.

`n_components` has to be valid for the wrapped estimator and the tensor. For PCA it cannot exceed the number of rows or the feature width.

### 2.2. What each estimator can do

| Estimator | What the fit uses | New rows |
| --- | --- | --- |
| `sklearn.decomposition.PCA` | Orthogonal directions of variance in the fitted rows | `transform` applies that fitted projection |
| `sklearn.random_projection.GaussianRandomProjection` | A Gaussian matrix drawn for the fitted feature width | `transform` applies that matrix |
| `umap.UMAP` | A nonlinear embedding that emphasizes local neighborhoods | `transform` embeds new rows into the fitted model |
| `sklearn.manifold.TSNE` | A nonlinear embedding of the fitted rows for visualization | No `transform` method |

PCA and Gaussian random projection are linear. They are the usual choice when the goal is a smaller input to selection or clustering. UMAP can preserve local structure that a linear map washes out, and it is heavier to fit. t-SNE is aimed at a plot of the rows in the fit. Its embedding is not a feature transform for a later batch.

Default PCA on a small dense matrix uses a deterministic solver and may ignore `random_state`. Randomized PCA, Gaussian random projection, UMAP, and t-SNE use randomness. See the reproducibility section for what Goldener can and cannot replay.

### 2.3. Fit, transform, and the scikit-learn adapter

`fit` learns the estimator and returns nothing. `fit_transform` learns it and returns the reduced rows of that same tensor. For an estimator that implements `transform`, a later `transform` uses the learned mapping and does not fit again. Row `i` of a PCA result is the projection of input row `i`. `transform` of the fitted tensor agrees with `fit_transform` up to ordinary float32 roundoff from the NumPy conversion.

On every `fit`, `fit_transform`, and successful `transform`, `GoldSKLearnReductionTool` detaches the tensor, moves it to CPU, and converts it with NumPy. The scikit-learn or UMAP call runs there. The adapter then builds a tensor with `torch.from_numpy(...).to(device=x.device, dtype=x.dtype)`. float32 and float64 inputs come back with that dtype and on the input device. bfloat16 has no NumPy conversion, so that input raises. The adapter does not leave a supported result on CPU unless the input was already there.

The number of rows is unchanged. The feature width becomes the estimator's `n_components`, or whatever width that estimator returns.

`GoldTorchModuleReductionTool` does not use this CPU conversion. It validates the tensor and calls the module. The module's device and dtype rules apply.

### 2.4. t-SNE has no `transform`

`TSNE` has no `transform` method. `GoldSKLearnReductionTool` records that at initialization. A later `transform` logs this warning and calls `fit_transform` on the tensor it was given:

`The provided reducer does not have a 'transform' method. Falling back to 'fit_transform' for transformation.`

That call embeds those rows from scratch. It does not place them into the coordinate system of an earlier fit. The fitted estimator is replaced by the new fit.

```python
from sklearn.manifold import TSNE

tsne_tool = GoldSKLearnReductionTool(
    TSNE(n_components=2, perplexity=5, random_state=0),
)
embedded = tsne_tool.fit_transform(x)
print(embedded.shape)  # (50, 2), an embedding of all 50 rows

# This warns, then embeds these 12 rows on their own.
refit = tsne_tool.transform(x[:12])
print(refit.shape)  # (12, 2)
```

`perplexity` must be smaller than the number of rows in that call. `GoldSelector` and `GoldClusterizer` do not go through this fallback for t-SNE: t-SNE is a `GoldReductionToolWithFit`, so they call `fit_transform` on each population they process.

### 2.5. Reproducibility

`GoldSKLearnReductionTool` can own the estimator's NumPy `RandomState`. Pass `random_state` as an integer or a `numpy.random.RandomState` to override the estimator. An integer becomes a `RandomState`, because a seed is only the initial state and cannot restore a generator that has already advanced. When the argument is omitted, a `RandomState` already stored on the estimator is reused. Any other estimator setting, including an integer `random_state` left on the estimator, is left untouched, and the tool's `random_state` property returns `None`.

`random_state` returns an independent snapshot from `RandomState.get_state(legacy=False)`. Reading it does not draw random numbers. `update_random_state` restores a snapshot. The next `fit` or `fit_transform` can then replay that estimator's random choices on the same input and configuration.

```python
from sklearn.random_projection import GaussianRandomProjection

projection = GoldSKLearnReductionTool(
    GaussianRandomProjection(n_components=4),
    random_state=0,
)
snapshot = projection.random_state
first = projection.fit_transform(x)
projection.update_random_state(snapshot)
replayed = projection.fit_transform(x)
print(torch.allclose(first, replayed))
```

The snapshot belongs to this tool and this sequence of calls. Another fit is another mapping. A second population, a second chunk, or a second label is fit on its own rows. Using the same seed for that other fit does not put both results in one coordinate system, and restored randomness does not make those projections interchangeable. UMAP treats a fixed `random_state` as incompatible with a parallel `n_jobs` and warns when it overrides that setting. That warning comes from UMAP.

Chunk assignment on `GoldSelector` and `GoldClusterizer` uses the doer's own integer `random_state`. That seed is not the reducer's snapshot.

### 2.6. A torch module

`GoldTorchModuleReductionTool` is the path for a mapping that is already a module, such as a frozen linear layer. There is no `fit`. The module must accept `(num_vectors, feature_dim)` and return one row per input row, or downstream index alignment is lost. Match the module's device to the tensor; this adapter does not move data to CPU or restore dtype.

```python
from torch.nn import Linear

from goldener import GoldTorchModuleReductionTool

linear = Linear(in_features=16, out_features=4)
module_tool = GoldTorchModuleReductionTool(linear)
module_out = module_tool.transform(x)
print(module_out.shape)  # (50, 4)
```

`GoldSelector` and `GoldClusterizer` call `transform` for a tool that is not a `GoldReductionToolWithFit`. A module is applied as it is, including to every label and chunk, without being fit on that population.

## 3. Selection and clustering

Pass the tool as `reducer` to `GoldClusterizer` or `GoldSelector`. The doer still clusters or selects in its usual way. Before each fit of the downstream tool, it reduces the vectors for that call.

For a `GoldReductionToolWithFit` the doer calls `fit_transform` on the rows in that call. For a transform-only tool it calls `transform`. The same reducer object is reused, so each `fit_transform` replaces the previous fit. After a chunked run, the in-memory estimator is the fit from the last population, not a fit of the full dataset.

With `label_key` set, clustering and selection each take a separate pass per distinct label value. The reducer's fit stays inside that pass. Cluster numbers and reduced coordinates from one label are not a coordinate system for another label.

With `chunk` set, each pass is split into random chunks of at most that many vectors. `chunk=None`, the default, keeps the pass in one chunk when the population fits. Each chunk is reduced and then clustered or selected on its own. Cluster index `0` in one chunk is not the same group as cluster index `0` in another chunk. The same is true of the reduced coordinates: each fittable chunk learns its own map.

Selection has one extra input to that fit. When some vectors for the current value and label were already selected, `GoldSelector` loads those vectors in the original feature width, concatenates them in front of the current chunk, and `fit_transform`s the concatenation. It then splits the reduced rows back into anchors and candidates. The next chunk refits, again from original-space vectors. Anchors are there so the selection tool can see them in the same fit as the candidates. They are not pushed through an older mapping.

Reduced vectors are not written to a column for reuse. The stored result is still the cluster id or the selection value, joined back to `idx` and `idx_vector`.

`GoldDensityWeigher` accepts the same kind of reducer and fits a fittable reducer independently on each weighing chunk. See [Weighting training samples by embedding density](GOLD_DENSITY_WEIGHING.md).

The clusterizer below fits PCA on each label, and on each chunk inside that label, and then runs k-means in the reduced space. `get_cluster_indices(..., idx_key="idx")` returns sample ids. The default `idx_key` is `idx_vector`.

```python
from uuid import uuid4

from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from goldener import GoldClusterizer, GoldSKLearnClusteringTool


class VectorDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        return self._rows[index]


generator = torch.Generator().manual_seed(0)
cluster_vectors = torch.randn(24, 16, generator=generator)
cluster_rows = VectorDataset(
    [
        {
            "idx": pos,
            "vectorized": vector,
            "label": "class_a" if pos < 12 else "class_b",
        }
        for pos, vector in enumerate(cluster_vectors)
    ]
)
clusterizer = GoldClusterizer(
    table_path=f"reduction_cluster_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=3, n_init=1, random_state=0)
    ),
    reducer=GoldSKLearnReductionTool(PCA(n_components=2)),
    label_key="label",
    chunk=8,
    random_state=0,
)
clustered = clusterizer.cluster_in_table(cluster_rows, n_clusters=3)
class_a = GoldClusterizer.get_cluster_indices(
    clustered,
    cluster_idx=0,
    label_key="label",
    label_value="class_a",
    idx_key="idx",
)
print(sorted(class_a))
```

Each class has 12 vectors and `chunk=8`, so each class is split into two chunks. Each chunk gets its own PCA fit and its own k-means fit. The ids printed for cluster `0` are `class_a` sample ids labeled `0` in one of that label's chunks. The same cluster number in the other chunk is a separate group.

The selector is given its own PCA reducer and the same 24 vectors, without labels. This call fits PCA once, because there is no `label_key` and no `chunk`, and then selects four sample ids.

```python
from goldener import GoldGreedyKCenterSelectionTool, GoldSelector

selector_rows = VectorDataset(
    [
        {"idx": pos, "vectorized": vector}
        for pos, vector in enumerate(cluster_vectors)
    ]
)
selector = GoldSelector(
    table_path=f"reduction_select_{uuid4().hex}",
    selection_tool=GoldGreedyKCenterSelectionTool(device="cpu"),
    reducer=GoldSKLearnReductionTool(PCA(n_components=2)),
)
selected = selector.select_in_table(selector_rows, select_size=4, value="review")
review_ids = GoldSelector.get_selection_indices(
    selected,
    value="review",
    idx_key="idx",
)
print(sorted(review_ids))
```

`select_size` is the overall count, or a ratio in `(0, 1]`. With `label_key` set, that target is split across labels, and each label's chunks fit the reducer as described above. `distribute=True` is not implemented for these doers.

## 4. Bibliography

The adapters wrap these methods. The references describe the algorithms, not a benchmark of Goldener:

- Jolliffe, I. T., and Cadima, J. [Principal component analysis: a review and recent developments](https://doi.org/10.1098/rsta.2015.0202). Philosophical Transactions of the Royal Society A, 2016.
- Dasgupta, S., and Gupta, A. [An elementary proof of a theorem of Johnson and Lindenstrauss](https://doi.org/10.1002/rsa.10073). Random Structures and Algorithms, 2003. Background for random projection.
- McInnes, L., Healy, J., and Melville, J. [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426). arXiv:1802.03426, 2018.
- van der Maaten, L., and Hinton, G. [Visualizing Data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html). Journal of Machine Learning Research, 2008.
