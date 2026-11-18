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

Imbalance can exist both between classes and within them. Even when class counts are equal, some groups of similar samples can be much more frequent than others. Uniform sampling follows those frequencies, giving common patterns more exposure while underrepresented patterns contribute less often.

Embedding neighborhoods offer another view of the training distribution. These embeddings can come from pretrained foundation models, providing semantic representations without training a dedicated model from scratch. When a representation captures relevant semantics, many nearby vectors suggest a well-represented pattern; larger distances to neighbors suggest a less represented one. Density-based weights let training put more emphasis on the latter without removing the original data.

This complements class-level reweighting. For example, [Cui et al. (2019)](https://arxiv.org/abs/1901.05555) address imbalance using the effective number of samples per class. Goldener's density score instead depends on local geometry and can vary between samples with the same label.

## 2. Components

### 2.1. From samples to vectors

The starting point for sample weighting is a vector representation. `GoldDescriptor` extracts embeddings, and `GoldVectorizer` prepares vectors when a separate vectorization stage is needed. A descriptor configured with a vectorizer can already produce suitable vectors.

`GoldDensityWeigher` accepts either a PyTorch `Dataset` (including a list of dictionaries as above) or a PixelTable `Table`. Dataset items provide a finite floating one-dimensional tensor under `vectorized_key`, which defaults to `"vectorized"`. A table provides the vector column together with `idx` and `idx_vector`. Set `vectorized_key` explicitly when an upstream stage uses another column name.

`idx` identifies the source sample; `idx_vector` uniquely identifies a vector. A sample can have several vectors representing different parts or aspects of its content. Supply the same `idx` on those rows and a different `idx_vector` on each. Dataset inputs without IDs receive vector IDs from item positions, and missing sample IDs default to those vector IDs; explicit IDs avoid losing an existing sample association.

The following examples reuse the imports and toy embeddings from the TLDR. Here, two vectors describe each sample; retrieval produces four sample weights from eight vector scores.

```python
multi_vector_input = [
    {"idx": sample_ids[pos // 2], "idx_vector": pos, "vectorized": vector}
    for pos, vector in enumerate(embeddings)
]
multi_weigher = GoldDensityWeigher(
    table_path=f"density_multi_{uuid4().hex}",
    weigher=GoldKnnDensityWeighingTool(k=2),
)
multi_weighted = multi_weigher.weigh_in_table(multi_vector_input)
multi_sample_weights = multi_weigher.get_weights_per_idx(multi_weighted)
print(multi_sample_weights)
```

### 2.2. What the kNN score measures

[Goldener](https://github.com/goldener-data/goldener) provides a default weighing algorithm implemented by [GoldKnnDensityWeighingTool](https://github.com/goldener-data/goldener/blob/main/goldener/weigh.py). However, the user can provide new weighing tools following the API defined in `GoldDensityWeighingTool`.

In `GoldKnnDensityWeighingTool`, for vector `x_i`, let `N_k(i)` contain its `k` nearest neighbors after excluding one occurrence of the query itself. Its score is:

$$
s_i = \left[\max\left(\frac{1}{k}\sum_{j \in N_k(i)}d(x_i,x_j),\ \varepsilon\right)\right]^p.
$$

The tool retrieves the nearest `k + 1` entries and discards one nearest entry. Other identical vectors remain valid neighbors. Defaults are `k=5`, Euclidean distance, `eps=1e-12`, and `power=0.5`. The current implementation requires **at least `k + 2` vectors in every scored population**, including every label-specific chunk.

For positive `power`, larger neighbor distances produce larger scores. The default square root compresses distance contrasts; increasing the exponent accentuates them. `power=0` produces uniform scores. The positive distance floor handles coincident vectors before exponentiation. This is a distance-based sparsity score, not a normalized probability density or a calibrated inverse-density estimate.

Choose the neighborhood metric through the tool's `distance` argument. `DistanceType.EUCLIDEAN` (the default) measures straight-line distance and reflects differences in both vector magnitude and direction. `DistanceType.COSINE` uses one minus cosine similarity, comparing vector directions after L2 normalization. For nonzero vectors, this makes cosine distance insensitive to magnitude, which is useful when embedding direction carries the relevant semantics. Import `DistanceType` from `goldener.select` and pass, for example, `distance=DistanceType.COSINE` to `GoldKnnDensityWeighingTool`. The metric changes neighbor rankings and score scales, so choose it to match the representation.

You can also score a tensor directly to compare metrics before storing results. These calls return raw scores in input row order.

```python
from goldener.select import DistanceType

# Use nonzero vectors so every row has a direction for cosine comparison.
metric_vectors = embeddings + 1.0
for metric in (DistanceType.EUCLIDEAN, DistanceType.COSINE):
    tool = GoldKnnDensityWeighingTool(k=2, distance=metric, power=0.5)
    scores = tool.weigh(metric_vectors)
    print(metric.value, scores)
```

### 2.3. Storage and sample aggregation

`GoldDensityWeigher.weigh_in_table` orchestrates ingestion and scoring, returning a local PixelTable table. Its default `weight` column contains **raw per-vector scores**. `weigh_in_dataset` offers the corresponding dataset wrapper; it also contains raw vector scores.

`get_weights_per_idx` converts those results into a dictionary. By default it:

1. Groups vector scores by sample `idx`.
2. Takes the arithmetic mean within each sample.
3. Divides each sample score by the maximum score in the returned dictionary.

For sample scores `a_i`, the returned weights are `w_i = a_i / max_j(a_j)`. All-zero scores remain zero. This is **max normalization, not mean-one normalization**, and retrieval does not modify persisted scores.

For samples represented by multiple vectors, averaging prevents vector count alone from directly summing into a larger sample weight. Vector count still affects the neighborhood population. Choose another aggregation only when it matches the training objective:

```python
raw_sample_weights = weigher.get_weights_per_idx(weighted, normalize=False)
raw_vector_scores = weigher.get_weights_per_idx(
    weighted, idx_key="idx_vector", normalize=False,
)
max_sample_weights = weigher.get_weights_per_idx(weighted, aggregation=max)
```

The last call uses each sample's highest vector score instead of its average. Retrieval requires every vector in the table to have a score, even when requesting a label-filtered result.

A few very isolated vectors can produce extreme raw scores, and max normalization then pushes most other weights close to zero. Set `max_raw_score` to cap the returned raw scores before normalization:

```python
capped_sample_weights = weigher.get_weights_per_idx(weighted, max_raw_score=5.0)
```

Clipping is disabled by default (`max_raw_score=None`). When set, it must be finite and positive. It is applied after sample aggregation (or directly to vector scores with `idx_key="idx_vector"`) and before normalization: scores above the cap are set to it, and scores at or below it are unchanged. With `normalize=True`, the cap therefore becomes the normalization reference whenever at least one score exceeds it. Like normalization, clipping only affects the returned dictionary, not the persisted scores.

### 2.4. Label populations, memory, and restart behavior

With `label_key=None` (the default), vectors share a population. Setting `label_key="label"` on the weigher scores each label population independently. Provide the corresponding label on input rows. This measures rarity within a class; it does not automatically equalize class influence. Multi-label samples can be represented by separate vector rows with unique vector IDs and a shared sample ID.

Retrieval can filter a label using `label_key="label", label_value="class_a"`. Aggregation and max normalization then use only returned rows. Consequently, weights retrieved for different filters need not share a common scale.

Each label below has four vectors, satisfying the `k + 2` minimum with `k=2`. Labels determine the neighbor populations during scoring; the retrieval filter then selects which results to return.

```python
labeled_input = [
    {**row, "label": "class_a" if pos < 4 else "class_b"}
    for pos, row in enumerate(vectorized)
]
label_weigher = GoldDensityWeigher(
    table_path=f"density_labels_{uuid4().hex}",
    weigher=GoldKnnDensityWeighingTool(k=2),
    label_key="label",
)
label_weighted = label_weigher.weigh_in_table(labeled_input)
class_a_weights = label_weigher.get_weights_per_idx(
    label_weighted, label_key="label", label_value="class_a",
)
print(class_a_weights)
```

There are three distinct controls for batching and chunking:

| Setting | What it controls | Neighbor population |
| --- | --- | --- |
| `GoldDensityWeigher.batch_size` | Dataset ingestion through the DataLoader | Unchanged by ingestion batching on a complete run |
| `GoldDensityWeigher.weighing_batch_size` and the tool's `distance_batch_size` | Query batching and pairwise-distance memory | Each query still sees its full chunk |
| `GoldDensityWeigher.chunk` | Random partition into independently scored chunks | Restricted to each chunk; scores can change |

Without chunking, exact pairwise search has quadratic distance work in the population size. Query batching reduces the temporary distance matrix size but still requires the reference vectors in memory. Chunking reduces the reference population at the cost of changing the score's meaning. Every resulting chunk must satisfy the `k + 2` minimum; a small remainder can make a configuration invalid. `random_state` controls random chunk assignment. An optional `reducer` transforms vectors before scoring; fittable reducers fit independently per chunk, also changing the geometry.

For a larger population of 20,000 vectors, the following configuration creates two independent chunks of 10,000 vectors. Within each chunk, queries are scored 256 at a time against all 10,000 reference vectors. Set `chunk=None` to retain a single reference population while keeping query batching.

```python
large_embeddings = torch.randn(
    20_000, 16, generator=torch.Generator().manual_seed(42),
)
large_vectorized = [
    {"idx": pos, "idx_vector": pos, "vectorized": vector}
    for pos, vector in enumerate(large_embeddings)
]
chunk_weigher = GoldDensityWeigher(
    table_path=f"density_chunks_{uuid4().hex}",
    weigher=GoldKnnDensityWeighingTool(k=5, distance_batch_size=256),
    batch_size=1_024,
    weighing_batch_size=256,
    chunk=10_000,
    random_state=42,
)
chunk_weighted = chunk_weigher.weigh_in_table(large_vectorized)
chunk_weights = chunk_weigher.get_weights_per_idx(chunk_weighted)
```

Completed raw scores are reused on repeated calls, without duplicate rows, provided IDs, inputs, and scoring settings remain unchanged. **Restart is not equivalent to recomputing a fixed full population.** Only pending vectors participate in new chunk assignments and neighbor searches; completed vectors are excluded. Chunk assignments are not persisted. After an interrupted run, remaining vectors can therefore receive different scores from an uninterrupted run, or fail the minimum population size check.

## 3. Density-aware training

The runnable example uses a `WeightedRandomSampler` to draw dataset positions with probability proportional to their aligned sample weights. Replacement permits rare samples to appear multiple times in an epoch.

An alternative is to keep uniform sampling and multiply each unreduced sample loss by its retrieved weight. For a batch of losses `loss_i`, a scale-controlled weighted average is `sum(w_i * loss_i) / sum(w_i)` when the denominator is positive. Using both weighted sampling and weighted loss compounds the emphasis, so choose that combination deliberately.

This minimal training step uses uniform sampling and an unreduced cross-entropy loss. It gathers weights by the batch's sample IDs, so shuffling preserves the association between samples and weights.

```python
model = torch.nn.Linear(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
uniform_loader = DataLoader(
    training_dataset,
    batch_size=4,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)
for batch in uniform_loader:
    logits = model(batch["data"])
    losses = torch.nn.functional.cross_entropy(
        logits, batch["target"], reduction="none",
    )
    batch_weights = losses.new_tensor(
        [sample_weights[int(idx)] for idx in batch["idx"]]
    )
    loss = (batch_weights * losses).sum() / batch_weights.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The default kNN tool produces positive weights for this example. With a custom tool that can return all-zero weights, skip batches whose total weight is zero before dividing.

Density scores can also inform a curriculum, for example by gradually increasing the exposure of sparse regions. Goldener supplies the scores; the training loop defines the schedule. [Hacohen and Weinshall (2019)](https://proceedings.mlr.press/v97/hacohen19a.html) study curriculum learning and its dependence on ordering and pacing.

## 4. Bibliography

Related reading, including training-prioritization work collected in [Goldener's open research bibliography](https://github.com/goldener-data/goldener-research/blob/main/training_strategy/BIBLIOGRAPHY.md):

- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., and Belongie, S. [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555). CVPR, 2019. Class-level reweighting as a complementary approach to imbalance.
- Mindermann, S., et al. [Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt](https://proceedings.mlr.press/v162/mindermann22a.html). ICML, 2022. Training prioritization based on learning utility, distinct from local embedding sparsity.
- Hacohen, G., and Weinshall, D. [On The Power of Curriculum Learning in Training Deep Networks](https://proceedings.mlr.press/v97/hacohen19a.html). ICML, 2019. Context for ordering and pacing training examples.

These papers provide context for reweighting and training policy design; they are not benchmark evidence for Goldener's kNN scoring implementation.
