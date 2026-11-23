# Reweight training samples by embedding density

## TL;DR

Use `GoldDensityWeigher` to give samples in sparse embedding neighborhoods more influence. Look up weights by sample ID and use them for sampling or loss weighting.

## Table of contents

- [Sample shipments by their density within a class](#sample-shipments-by-their-density-within-a-class)
- [Weight an image classification loss by sample ID](#weight-an-image-classification-loss-by-sample-id)
- [How scores are computed](#how-scores-are-computed)
- [Retrieve sample weights or vector scores](#retrieve-sample-weights-or-vector-scores)
- [Balance memory use and neighborhood scope](#balance-memory-use-and-neighborhood-scope)

Two examples can carry the same label even when one reflects a much less common case. In a table of delayed shipments, most delayed examples might be from busy routes. A model still needs to learn about delays on less traveled routes. If those routes form sparse neighborhoods in the feature space, local distances offer a way to give them more influence without changing the label.

`GoldDensityWeigher` scores each vector by how far it sits from nearby vectors. Use those scores to draw some samples more often or to give them more weight in the training loss.

## Sample shipments by their density within a class

These eight synthetic vectors stand in for delayed shipments. Six sit close together in feature space, and two sit apart. The k-nearest-neighbor tool scores their vectors, and the weigher looks up the resulting sample weights by ID. The sampler needs weights in the same order as the training dataset.

```python
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from goldener import GoldDensityWeigher, GoldKnnDensityWeighingTool

vectors = torch.tensor(
    [
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
        [0.2, 0.0], [0.0, 0.2], [3.0, 3.0], [5.0, 3.0],
    ],
)
sample_ids = [101, 205, 309, 412, 518, 623, 731, 845]
training_data = [
    {"idx": sample_id, "idx_vector": pos, "vectorized": vector, "label": "same_label"}
    for pos, (sample_id, vector) in enumerate(zip(sample_ids, vectors))
]

weigher = GoldDensityWeigher(
    table_path=f"training_weights_{uuid4().hex}",
    weigher=GoldKnnDensityWeighingTool(k=2),
    label_key="label",
)
table = weigher.weigh_in_table(training_data)
sample_weights = weigher.get_weights_per_idx(table)
weights_in_order = [sample_weights[row["idx"]] for row in training_data]

assert min(weights_in_order[-2:]) > max(weights_in_order[:6])
sampler = WeightedRandomSampler(
    weights_in_order,
    num_samples=len(training_data),
    replacement=True,
    generator=torch.Generator().manual_seed(42),
)
loader = DataLoader(training_data, batch_size=4, sampler=sampler)
for batch in loader:
    print(batch["idx"].tolist())
```

The two isolated samples have larger weights, so each has a higher probability per draw. Sampling with replacement can repeat some examples and leave others undrawn in an epoch.

## Weight an image classification loss by sample ID

A camera-trap model can keep its usual sampler and give uncommon images more influence in the loss. The example uses two synthetic region vectors per image. Set `exclude_same_sample=True` so the other region of a query image does not count as a neighbor. This makes the score reflect distance to regions from other images.

```python
from uuid import uuid4

import torch
from torch.nn.functional import cross_entropy

from goldener import GoldDensityWeigher, GoldKnnDensityWeighingTool

regions = torch.tensor(
    [[0.0, 0.0], [0.05, 0.0], [0.2, 0.0], [0.25, 0.0], [4.0, 4.0], [4.05, 4.0]],
)
image_ids = [11, 11, 22, 22, 33, 33]
rows = [
    {"idx": image_id, "idx_vector": pos, "vectorized": vector}
    for pos, (image_id, vector) in enumerate(zip(image_ids, regions))
]
weigher = GoldDensityWeigher(
    table_path=f"image_loss_weights_{uuid4().hex}",
    weigher=GoldKnnDensityWeighingTool(k=2),
    exclude_same_sample=True,
)
sample_weights = weigher.get_weights_per_idx(weigher.weigh_in_table(rows))

# Stand-in for logits, targets, and image IDs from an ordinary training batch.
batch_ids = [11, 33]
logits = torch.tensor([[2.0, -1.0], [2.0, -1.0]], requires_grad=True)
targets = torch.tensor([0, 1])
losses = cross_entropy(logits, targets, reduction="none")
batch_weights = losses.new_tensor([sample_weights[idx] for idx in batch_ids])
total_weight = batch_weights.sum()

assert sample_weights[33] > sample_weights[11]
if total_weight > 0:
    loss = (batch_weights * losses).sum() / total_weight
    assert loss > losses.mean()
    loss.backward()
    print(round(loss.item(), 3))
```

The two losses correspond to different images. Image 33 sits in a sparse neighborhood and receives more weight. The ID lookup keeps weights aligned even when the loader shuffles the batch. A custom weigher can yield a batch with zero total weight. In a training loop, skip that batch before dividing or taking an optimizer step. Using both a weighted sampler and a weighted loss compounds the emphasis.

## How scores are computed

1. `GoldDensityWeigher` reads vectors from a dataset or Pixeltable table and defines the reference population for each query. Set `label_key` to look for neighbors within a class, as in the shipment example. Per-label neighborhoods can change weights within a class. Their effects on class balance also depend on the scores and training policy. [Cui et al. on reweighting by class frequency](https://arxiv.org/abs/1901.05555)
2. `GoldKnnDensityWeighingTool` computes pairwise distances and discards ineligible references. When `exclude_same_sample=True`, the weigher passes sample IDs to the tool, which discards every reference with the query's `idx`. This requires at least `k + 1` reference rows and `k` eligible vectors from other samples for every query. Without sample IDs it drops one nearest reference and requires at least `k + 2` rows.
3. The tool averages the `k` nearest eligible distances, floors the mean at a small positive value, and raises it to `power`. The default square root softens distance differences.

Euclidean distance can reflect magnitude and direction. Cosine distance compares directions, which is useful when vector magnitude carries little meaning. Inspect highly weighted examples before using them. Sparse image regions could capture uncommon night sightings or blank frames from a camera fault.

## Retrieve sample weights or vector scores

The table stores one raw score per vector. By default, `get_weights_per_idx` averages the vector scores for each sample, then divides each sample score by the largest one if it is positive. To inspect the original vector scores, use the `table` from the shipment example.

```python
raw_vector_scores = GoldDensityWeigher.get_weights_per_idx(
    table, idx_key="idx_vector", normalize=False,
)
```

Filtering with `label_key` and `label_value` recalculates the normalization maximum from the filtered scores. Weights fetched separately for different labels can therefore have different scales. Set `max_raw_score` to cap high scores after sample aggregation and before normalization.

## Balance memory use and neighborhood scope

Set `weighing_batch_size` to score fewer queries at once using the same reference population. Exact search still does pairwise work that grows quadratically with population size. Set `chunk` to give each query a smaller reference population, which can change its score. The minimum neighbor count applies within every scored chunk or label group.

On resume, Goldener reuses saved scores and scores pending vectors against other pending vectors. A small pending group can fail the neighbor requirement. Use a fresh result table when you need to recompute scores over the full population. For stored embeddings, configure `vectorized_key` to match the vector column in your source dataset or Pixeltable table.

To control which modes appear together in each training batch, use a cluster-balanced batch sampler. [Building batches from embedding clusters](GOLD_BATCHER.md)
