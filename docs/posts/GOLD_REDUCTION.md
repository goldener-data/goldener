# GOLDENER <br> Reducing the dimensionality of embeddings

[**Introduction**](#1-introduction) |
[**Components**](#2-components) |
[**Bibliography**](#3-bibliography)

## TLDR

High-dimensional embeddings make processing algorithm expensive in memory and time. [Goldener](https://github.com/goldener-data/goldener) can project those vectors with `GoldSKLearnReductionTool` or `GoldTorchModuleReductionTool` before the downstream computation. A fitted linear method such as PCA can then place new rows in the same mapping. The projection drops information, and a visualization method such as t-SNE does not provide that reusable mapping.

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

Embeddings from a foundation model are often hundreds or thousands of dimensions wide. A processing step on those vectors spends memory and time that grow with the feature width, so a smaller representation can make a later pass fit in memory and finish sooner. The reducer itself has a cost. The saving appears when that cost is smaller than the work it replaces.

A shorter vector is a lossy view of the original embedding. Directions that PCA treats as low variance, or neighbors that a nonlinear method fails to preserve, can be exactly the differences a later step needed. Check a sample of original neighbors before treating a reduced run as equivalent to the full embedding.

## 2. Components

### 2.1. Tensor interfaces

Reduction operates on vectors that are already 2D. Every tool expects a finite floating tensor shaped `(num_vectors, feature_dim)` with a positive feature width. Row `i` is one vector. The four public types are exported from `goldener`:

- `GoldReductionTool` is the abstract interface. Subclasses implement `transform(x)`.
- `GoldReductionToolWithFit` extends that interface with `fit(x)` and `fit_transform(x)` for methods that learn from data.
- `GoldSKLearnReductionTool` is the fittable adapter for UMAP, PCA, t-SNE, and Gaussian random projection.
- `GoldTorchModuleReductionTool` wraps a `torch.nn.Module`. It only implements `transform`; it does not fit the module.

UMAP, PCA, t-SNE, and Gaussian random projection are the supported estimators. The constructor annotation names those four, and the constructor stores the given object without checking its type. Passing one of them does not mean the four algorithms share a transform, a random-state contract, or a comparable geometry. A tensor that is not 2D raises `ValueError` starting with `GoldSKLearnReductionTool only accepts 2D tensors`. A feature width of zero raises `ValueError` with `GoldSKLearnReductionTool requires positive feature width.` A tensor that is not floating, or that contains a nonfinite value, raises `ValueError` with `x must contain only finite floating point values.` That message does not name the tool.

`n_components` has to be valid for the wrapped estimator and the tensor. For PCA it cannot exceed the number of rows or the feature width.

### 2.2. What each estimator can do

| Estimator | What the fit uses | New rows |
| --- | --- | --- |
| `sklearn.decomposition.PCA` | Orthogonal directions of variance in the fitted rows | `transform` applies that fitted projection |
| `sklearn.random_projection.GaussianRandomProjection` | A Gaussian matrix drawn for the fitted feature width | `transform` applies that matrix |
| `umap.UMAP` | A nonlinear embedding that emphasizes local neighborhoods | `transform` embeds new rows into the fitted model |
| `sklearn.manifold.TSNE` | A nonlinear embedding of the fitted rows for visualization | No `transform` method |

PCA and Gaussian random projection are linear. They are the usual choice when the goal is a smaller feature width and a `transform` for later rows. UMAP can preserve local structure that a linear map washes out, and it is heavier to fit. t-SNE is aimed at a plot of the rows in the fit. Its embedding is not a feature transform for a later batch.

Default PCA on a small dense matrix uses a deterministic solver and may ignore `random_state`. Randomized PCA, Gaussian random projection, UMAP, and t-SNE use randomness. See the reproducibility section for what Goldener can and cannot replay.

### 2.3. Fit, transform, and the scikit-learn adapter

`fit` learns the estimator and returns nothing. `fit_transform` learns it and returns the reduced rows of that same tensor. A later `fit` or `fit_transform` on the same tool replaces the fitted estimator. For an estimator that implements `transform`, a later `transform` uses the learned mapping and does not fit again. A tool that is not a `GoldReductionToolWithFit` has no `fit`; `transform` applies the mapping it already has. Row `i` of a PCA result is the projection of input row `i`. `transform` of the fitted tensor agrees with `fit_transform` up to ordinary float32 roundoff from the NumPy conversion.

After validation, `fit`, `fit_transform`, and `transform` detach the tensor, move it to CPU, and convert it with NumPy before the scikit-learn or UMAP call. `fit` returns nothing. `fit_transform` and `transform` convert the estimator output with `torch.from_numpy(...).to(device=x.device, dtype=x.dtype)`. float32 and float64 results keep that dtype and the input device. bfloat16 has no NumPy conversion, so that input raises. The adapter does not leave a supported result on CPU unless the input was already there.

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

`perplexity` must be smaller than the number of rows in that call. t-SNE is a `GoldReductionToolWithFit`, so `fit_transform` is the call that learns an embedding of those rows. `transform` is the fallback above, and it replaces the fitted estimator.

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

The snapshot belongs to this tool and this sequence of calls. PCA, UMAP, and t-SNE learn their map from the rows in the fit. A second fit on different rows gets a different map, including when both fits start from the same random state. Gaussian random projection with an integer `n_components` draws its matrix from the random state and the feature width, then stores that matrix in the input dtype. The values in the rows are not part of that draw. Two fits that begin from the same random state, with the same feature width, the same `n_components`, and the same dtype, rebuild that matrix, so both results lie in one coordinate system. Each fit still projects its own rows. A tool that owns the `RandomState` begins there again only when `update_random_state` restores the snapshot. Its next fit otherwise draws a new matrix. A later `fit` or `fit_transform` on that tool leaves the advanced state in place. An integer `random_state` left on the estimator restarts from that integer on every fit, so those refits share the matrix when the feature width, `n_components`, and dtype match. With `n_components="auto"`, the width also depends on the number of rows, so fits on tensors with different numbers of rows can draw different matrices. UMAP treats a fixed `random_state` as incompatible with a parallel `n_jobs` and warns when it overrides that setting. That warning comes from UMAP.

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

## 3. Bibliography

The adapters wrap these methods. The references describe the algorithms, not a benchmark of Goldener:

- Jolliffe, I. T., and Cadima, J. [Principal component analysis: a review and recent developments](https://doi.org/10.1098/rsta.2015.0202). Philosophical Transactions of the Royal Society A, 2016.
- Dasgupta, S., and Gupta, A. [An elementary proof of a theorem of Johnson and Lindenstrauss](https://doi.org/10.1002/rsa.10073). Random Structures and Algorithms, 2003. Background for random projection.
- McInnes, L., Healy, J., and Melville, J. [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426). arXiv:1802.03426, 2018.
- van der Maaten, L., and Hinton, G. [Visualizing Data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html). Journal of Machine Learning Research, 2008.
