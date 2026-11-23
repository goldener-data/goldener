# Put different kinds of examples in each training batch

## TL;DR

Use `GoldClusterizedBatchSampler` to draw from several embedding clusters in each training batch. The exhaustion policy determines whether small clusters repeat, training stops early, or later batches draw from fewer clusters.

## Table of contents

- [Mix acoustic conditions in each batch](#mix-acoustic-conditions-in-each-batch)
- [Decide what happens when a pool runs out](#decide-what-happens-when-a-pool-runs-out)
- [Use every event session without recycling a small group](#use-every-event-session-without-recycling-a-small-group)
- [How it works](#how-it-works)

Shuffling mixes a training set over an epoch, but many batches can still be dominated by a common pattern. A speech command model trained mostly on quiet recordings may see background conversation in very few updates. If embeddings separate those acoustic conditions, drawing across clusters can expose the model to both more regularly.

`GoldClusterizedBatchSampler` groups samples by embedding similarity, then draws across the resulting pools. The cluster count determines how many pools compete for the available batch positions.

## Mix acoustic conditions in each batch

These synthetic audio vectors stand in for four quiet recordings and two recordings with background conversation. The `mode` field lets us inspect batch composition. The k-means estimator uses only the vectors. With two clusters, a batch size of two, and `RESTART`, each batch will contain one sample from each group.

```python
from uuid import uuid4

import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from goldener import GoldClusterizedBatchSampler, GoldClusterizer, GoldSKLearnClusteringTool
from goldener.organize import ExhaustedClusterStrategy

vectors = torch.tensor(
    [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [5.0, 5.0], [5.1, 5.0]],
)
training_data = [
    {"idx": pos, "vectorized": vector, "mode": "A" if pos < 4 else "B"}
    for pos, vector in enumerate(vectors)
]
clusterizer = GoldClusterizer(
    table_path=f"batch_clusters_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(KMeans(random_state=0)),
)
batch_sampler = GoldClusterizedBatchSampler(
    dataset=training_data,
    batch_size=2,
    n_clusters=2,
    clusterizer=clusterizer,
    shuffle=False,
    strategy=ExhaustedClusterStrategy.RESTART,
)
loader = DataLoader(training_data, batch_sampler=batch_sampler)

seen = []
for batch in loader:
    assert sorted(batch["mode"]) == ["A", "B"]
    seen.extend(batch["idx"].tolist())
    print(batch["idx"].tolist(), batch["mode"])

assert len(seen) == 8
assert set(seen) == set(range(len(training_data)))
```

The epoch draws four batches, so each sample in group B appears twice and each sample in group A appears once. In training you would use the data and targets from each batch in the optimizer step. Pass the sampler as `batch_sampler` because it supplies complete index lists to the `DataLoader`.

## Decide what happens when a pool runs out

The three `ExhaustedClusterStrategy` policies make different trade-offs on those same two pools.

| Policy | What happens with four samples in A and two in B |
| --- | --- |
| `RESTART` (default) | Draw four mixed batches by cycling through group B a second time. Both modes appear throughout the epoch, and the smaller pool is oversampled. |
| `STOP` | Yield the second mixed batch, then stop because group B is exhausted. Two samples from group A go unused in that epoch. |
| `EXCLUDE` | Draw two mixed batches, then one batch of group A alone. Each sample is used once in this example, and the tail has less variety. Other pool sizes can leave a smaller final batch. |

Choose based on how much repetition you can tolerate and whether an epoch needs to reach every training sample. With more clusters than batch positions, each update sees a subset of the pools. With fewer clusters, the sampler can draw several examples from a pool within a batch. The example disables shuffling to make the draws easy to inspect. In training, shuffling changes the order of samples within pools and helps vary which pools appear when a batch can only fit some of them.

## Use every event session without recycling a small group

Suppose a session model trains on a subset of user event sequences. The synthetic vectors below use session length as one feature. The clusterer separates shorter and longer sessions. Use `EXCLUDE` to see each session once, even if the last batch comes from a single group. Keep each `idx` tied to its position in the original dataset.

```python
from uuid import uuid4

import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset

from goldener import GoldClusterizedBatchSampler, GoldClusterizer, GoldSKLearnClusteringTool
from goldener.organize import ExhaustedClusterStrategy

event_counts = [1, 2, 2, 3, 4, 14, 15, 16]
all_sessions = [
    {"idx": pos, "vectorized": torch.tensor([float(count), 1.0]), "event_count": count}
    for pos, count in enumerate(event_counts)
]
training = Subset(all_sessions, [1, 2, 3, 5, 6])
clusterizer = GoldClusterizer(
    table_path=f"session_clusters_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(KMeans(random_state=0)),
)
sampler = GoldClusterizedBatchSampler(
    dataset=training,
    batch_size=2,
    n_clusters=2,
    clusterizer=clusterizer,
    shuffle=False,
    strategy=ExhaustedClusterStrategy.EXCLUDE,
)
loader = DataLoader(training, batch_sampler=sampler)

batches = [batch["idx"].tolist() for batch in loader]
seen = [idx for batch in batches for idx in batch]
assert sorted(seen) == sorted(training.indices)
assert [len(batch) for batch in batches] == [2, 2, 1]
for batch in batches[:2]:
    shorter, longer = sorted(event_counts[idx] for idx in batch)
    assert shorter < 5 and longer >= 14
print(batches)
```

The first two batches mix shorter and longer sessions. Once the longer pool runs out, the remaining short session forms the last batch. The sampler supplies row positions. If your training items contain variable-length event sequences, use the loader's `collate_fn` to pad or pack them.

## How it works

1. The sampler runs the supplied `GoldDescriptor` and `GoldVectorizer` if configured, then calls the clusterizer on the training data. If the descriptor already produces one vector per item and you omit the separate vectorizer, set the clusterizer's `vectorized_key` to the descriptor's `description_key` (default `"embeddings"`). Use a vectorizer if the stored embeddings still need to become vectors. It reads each cluster's `idx` values into an index pool. Use one vector per item and a single fit over the training population to give each item one pool.
2. For a PyTorch `Subset`, the sampler maps base-dataset `idx` values to positions within the subset. Those subset positions are what the `DataLoader` uses to look up items. The example prints the original IDs carried by the items, so you can verify coverage against `training.indices`.
3. At iteration time, the sampler orders each pool, optionally shuffles it, and advances a cursor each time it draws from that pool. A pass draws at most one item per active cluster. The sampler can take more passes to fill a large batch, or spread the pools over several batches when there are fewer positions than pools. At exhaustion the chosen policy controls whether a cursor resets, the pool is removed, or iteration stops after the current batch. [How to inspect an embedding partition](GOLD_CLUSTERIZER.md)

Cluster balancing changes which examples meet in an optimizer update and, with `RESTART`, how often the smaller pools contribute. Embedding-density weights let you adjust a sample's draw frequency or loss contribution based on its local rarity. Compare the policies on the evaluation slices that matter to the model. [Weighting training samples by embedding density](GOLD_DENSITY_WEIGHING.md) · [Zhao and Zhang on stratified minibatch sampling](https://arxiv.org/abs/1405.3080)

Sponsored by [Pixeltable](https://www.pixeltable.com/).
