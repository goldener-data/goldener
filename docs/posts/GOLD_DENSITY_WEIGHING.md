# GOLDENER <br> Weighting training samples by embedding density

[**Introduction**](#1-introduction) |
[**Components**](#2-components) |
[**Density-aware training**](#3-density-aware-training) |
[**Bibliography**](#4-bibliography)

## TLDR

[Goldener](https://github.com/goldener-data/goldener) can give more influence to samples in sparse embedding neighborhoods. `GoldDensityWeigher` stores a raw score for each vector, then retrieves weights associated with sample IDs for loss weighting or sampling. This provides a way to address semantic imbalance, including variation hidden inside a single class.

The following example runs with Goldener installed, without downloading a model. The two-dimensional vectors stand in for embeddings already extracted from a pretrained network. Run it in a Python script or notebook; each run uses a fresh local PixelTable table.

```python
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from goldener import GoldDensityWeigher, GoldKnnDensityWeighingTool

# A dense group and a sparse group, all belonging to the same training class.
embeddings = torch.tensor(
    [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
     [0.2, 0.0], [0.0, 0.2], [3.0, 3.0], [5.0, 3.0]],
    dtype=torch.float32,
)
sample_ids = [101, 205, 309, 412, 518, 623, 731, 845]
vectorized = [
    {"idx": idx, "idx_vector": pos, "vectorized": vector}
    for pos, (idx, vector) in enumerate(zip(sample_ids, embeddings))
]

weigher = GoldDensityWeigher(
    table_path=f"density_demo_{uuid4().hex}",
    weigher=GoldKnnDensityWeighingTool(k=2, power=0.5),
)
weighted = weigher.weigh_in_table(vectorized)
sample_weights = weigher.get_weights_per_idx(weighted)
print(sample_weights)  # {sample idx: weight}, not a list in dataset order

# Align weights with the positions of samples in the training dataset.
training_dataset = [
    {"idx": idx, "data": vector, "target": 0}
    for idx, vector in zip(sample_ids, embeddings)
]
weights_in_training_order = torch.tensor(
    [sample_weights[sample["idx"]] for sample in training_dataset],
    dtype=torch.float64,
)
sampler = WeightedRandomSampler(
    weights=weights_in_training_order,
    num_samples=len(training_dataset),
    replacement=True,
    generator=torch.Generator().manual_seed(42),
)
loader = DataLoader(training_dataset, batch_size=4, sampler=sampler)
for batch in loader:
    print(batch["idx"])  # Replace with the training step.
```

The toy dataset illustrates the mechanics rather than a measured training improvement. Keep `weighted` to inspect the raw per-vector `weight` column; the retrieved dictionary defaults to sample averaging and max normalization.

## 1. Introduction

Training data rarely reflects exactly how we want a model to learn. Common cases can dominate the dataset, important situations may be scarce, and repeated examples can consume much of the training budget. Sample weighting lets us adjust how much each example contributes, either by changing how often it is sampled or by scaling its contribution to the loss. This makes it possible to emphasize underrepresented cases, reduce the influence of redundant data, or adapt training priorities over time without discarding samples.

Consider a visual inspection dataset with thousands of nearly identical views of a product and only a few examples under unusual lighting. Both groups can share the same class label. Class counts alone cannot reveal this imbalance: uniform sampling repeatedly presents the common appearance and rarely presents the unusual one.

Embedding neighborhoods offer another view of the training distribution. When a representation captures relevant semantics, many nearby vectors suggest a well-represented appearance; larger distances to neighbors suggest a less represented one. Density-based weights let training put more emphasis on the latter without removing the original data.

This complements class-level reweighting. For example, [Cui et al. (2019)](https://arxiv.org/abs/1901.05555) address imbalance using the effective number of samples per class. Goldener's density score instead depends on local geometry and can vary between samples with the same label.

**Sparsity is not calibrated sample difficulty.** The score does not inspect the current model's predictions, gradients, or loss. An isolated vector might represent a useful rare case, a mislabeled example, or an artifact. A dense region might still contain a difficult decision boundary. Methods such as [Mindermann et al. (2022)](https://proceedings.mlr.press/v162/mindermann22a.html) use learning-related signals to prioritize training; Goldener's kNN score is a representation-based signal with a different meaning.

Use density weighting when underrepresented semantic regions matter to the task. Inspect the highest-weight samples and compare against uniform training on a held-out evaluation set before adopting it.

## 2. Components

### 2.1. From samples to vectors

As in Goldener's [selection](GOLD_SELECTION.md) and [batching](GOLD_BATCHER.md) pipelines, the starting point is a vector representation. `GoldDescriptor` extracts embeddings, and `GoldVectorizer` prepares vectors when a separate vectorization stage is needed. A descriptor configured with a vectorizer can already produce suitable vectors.

`GoldDensityWeigher` accepts either a PyTorch `Dataset` (including a list of dictionaries as above) or a PixelTable `Table`. Dataset items provide a finite floating one-dimensional tensor under `vectorized_key`, which defaults to `"vectorized"`. A table provides the vector column together with `idx` and `idx_vector`. Set `vectorized_key` explicitly when an upstream stage uses another column name.

`idx` identifies the source sample; `idx_vector` uniquely identifies a vector. A sample can have several vectors, for example one per image patch or region. Supply the same `idx` on those rows and a different `idx_vector` on each. Dataset inputs without IDs receive vector IDs from item positions, and missing sample IDs default to those vector IDs; explicit IDs avoid losing an existing sample association.

### 2.2. What the kNN score measures

The tensor algorithm is implemented by [GoldKnnDensityWeighingTool](https://github.com/goldener-data/goldener/blob/main/goldener/weigh.py), an implementation of `GoldDensityWeighingTool`. For vector `x_i`, let `N_k(i)` contain its `k` nearest neighbors after excluding one occurrence of the query itself. Its score is:

$$
s_i = \left[\max\left(\frac{1}{k}\sum_{j \in N_k(i)}d(x_i,x_j),\ \varepsilon\right)\right]^p.
$$

The tool retrieves the nearest `k + 1` entries and discards one nearest entry. Other identical vectors remain valid neighbors. Defaults are `k=5`, Euclidean distance, `eps=1e-12`, and `power=0.5`. The current implementation requires **at least `k + 2` vectors in every scored population**, including every label-specific chunk.

For positive `power`, larger neighbor distances produce larger scores. The default square root compresses distance contrasts; increasing the exponent accentuates them. `power=0` produces uniform scores. The positive distance floor handles coincident vectors before exponentiation. This is a distance-based sparsity score, not a normalized probability density or a calibrated inverse-density estimate.

The deterministic tool accepts a two-dimensional tensor shaped `(num_vectors, feature_dim)` and returns one score per row. Its configurable distance metric and the scale of the embeddings define what counts as nearby, so representation choice and preprocessing matter.

### 2.3. Storage and sample aggregation

`GoldDensityWeigher.weigh_in_table` orchestrates ingestion and scoring, returning a local PixelTable table. Its default `weight` column contains **raw per-vector scores**. `weigh_in_dataset` offers the corresponding dataset wrapper; it also contains raw vector scores.

`get_weights_per_idx` converts those results into a dictionary. By default it:

1. Groups vector scores by sample `idx`.
2. Takes the arithmetic mean within each sample.
3. Divides each sample score by the maximum score in the returned dictionary.

For sample scores `a_i`, the returned weights are `w_i = a_i / max_j(a_j)` when the maximum is positive. All-zero scores remain zero. This is **max normalization, not mean-one normalization**, and retrieval does not modify persisted scores.

For samples represented by multiple vectors, averaging prevents vector count alone from directly summing into a larger sample weight. Vector count still affects the neighborhood population. Choose another aggregation only when it matches the training objective:

```python
raw_sample_weights = weigher.get_weights_per_idx(weighted, normalize=False)
raw_vector_scores = weigher.get_weights_per_idx(
    weighted, idx_key="idx_vector", normalize=False,
)
max_region_weights = weigher.get_weights_per_idx(weighted, aggregation=max)
```

The last call emphasizes each sample's sparsest region. Retrieval requires every vector in the table to have a score, even when requesting a label-filtered result.

### 2.4. Label populations, memory, and restart behavior

With `label_key=None` (the default), vectors share a population. Setting `label_key="label"` on the weigher scores each label population independently. Provide the corresponding label on input rows. This measures rarity within a class; it does not automatically equalize class influence. Multi-label samples can be represented by separate vector rows with unique vector IDs and a shared sample ID.

Retrieval can filter a label using `label_key="label", label_value="defect"`. Aggregation and max normalization then use only returned rows. Consequently, weights retrieved for different filters need not share a common scale.

There are three distinct controls for batching and chunking:

| Setting | What it controls | Neighbor population |
| --- | --- | --- |
| `GoldDensityWeigher.batch_size` | Dataset ingestion through the DataLoader | Unchanged by ingestion batching on a complete run |
| `GoldDensityWeigher.weighing_batch_size` and the tool's `distance_batch_size` | Query batching and pairwise-distance memory | Each query still sees its full chunk |
| `GoldDensityWeigher.chunk` | Random partition into independently scored chunks | Restricted to each chunk; scores can change |

Without chunking, exact pairwise search has quadratic distance work in the population size. Query batching reduces the temporary distance matrix size but still requires the reference vectors in memory. Chunking reduces the reference population at the cost of changing the score's meaning. Every resulting chunk must satisfy the `k + 2` minimum; a small remainder can make a configuration invalid. `random_state` controls random chunk assignment. An optional `reducer` transforms vectors before scoring; fittable reducers fit independently per chunk, also changing the geometry. Distributed execution is currently unsupported.

Completed raw scores are reused on repeated calls, without duplicate rows, provided IDs, inputs, and scoring settings remain unchanged. **Restart is not equivalent to recomputing a fixed full population.** Only pending vectors participate in new chunk assignments and neighbor searches; completed vectors are excluded. Chunk assignments are not persisted. After an interrupted run, remaining vectors can therefore receive different scores from an uninterrupted run, or fail the minimum population size check.

Similarly, `max_batches` limits the ingested source prefix on each call. Raising it adds rows without updating previous scores; new scores describe the new pending population. It is useful for exploration, but incremental ingestion does not maintain exact full-dataset density scores. Use a fresh result table and score the full intended population for a consistent comparison after changing data or settings. Retrieval normalization can also change when newly returned scores change the maximum.

## 3. Density-aware training

The runnable example uses a `WeightedRandomSampler` to draw dataset positions with probability proportional to their aligned sample weights. Replacement permits rare samples to appear multiple times in an epoch. The dictionary keys are sample IDs, which may be noncontiguous; the sampler needs weights in **training dataset order**. Sampling does not guarantee balanced classes or balanced individual mini-batches.

An alternative is to keep uniform sampling and multiply each unreduced sample loss by its retrieved weight. For a batch of losses `loss_i`, a scale-controlled weighted average is `sum(w_i * loss_i) / sum(w_i)` when the denominator is positive. Move the batch weights to the loss tensor's device and dtype. Simply computing `mean(w_i * loss_i)` also changes the loss magnitude because the default weights do not have mean one. Using both weighted sampling and weighted loss compounds the emphasis, so choose that combination deliberately.

Density scores can also inform a curriculum, for example by gradually increasing the exposure of sparse regions. Goldener supplies the scores; the training loop defines the schedule. [Hacohen and Weinshall (2019)](https://proceedings.mlr.press/v97/hacohen19a.html) study curriculum learning and its dependence on ordering and pacing. Their results do not establish that sparse embeddings are intrinsically harder or that this density-based schedule improves a particular task.

Evaluate the policy with the same data split and training budget as the uniform baseline. Compare class and subgroup metrics as well as aggregate accuracy, and inspect whether high weights concentrate on useful rare cases or on noise. The feature makes the weighting policy explicit and traceable to sample IDs; the appropriate policy remains task dependent.

## 4. Bibliography

Related reading, including training-prioritization work collected in [Goldener's open research bibliography](https://github.com/goldener-data/goldener-research/blob/main/training_strategy/BIBLIOGRAPHY.md):

- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., and Belongie, S. [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555). CVPR, 2019. Class-level reweighting as a complementary approach to imbalance.
- Mindermann, S., et al. [Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt](https://proceedings.mlr.press/v162/mindermann22a.html). ICML, 2022. Training prioritization based on learning utility, distinct from local embedding sparsity.
- Hacohen, G., and Weinshall, D. [On The Power of Curriculum Learning in Training Deep Networks](https://proceedings.mlr.press/v97/hacohen19a.html). ICML, 2019. Context for ordering and pacing training examples.

These papers provide context for reweighting and training policy design; they are not benchmark evidence for Goldener's kNN scoring implementation.
