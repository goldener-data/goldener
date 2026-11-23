# Put different kinds of examples in each training batch

## TL;DR

Use `GoldClusterizedBatchSampler` to draw from embedding clusters in each training batch. Choose whether to repeat smaller clusters, end the epoch early, or keep drawing from the clusters that still have samples.

## Table of contents

- [Mix chest X-ray acquisition settings in each batch](#mix-chest-x-ray-acquisition-settings-in-each-batch)
- [Decide what happens when a pool runs out](#decide-what-happens-when-a-pool-runs-out)
- [Use every event session without recycling a small group](#use-every-event-session-without-recycling-a-small-group)
- [Prepare vectors and dataset indices](#prepare-vectors-and-dataset-indices)

Shuffling mixes a training set over an epoch, but many batches can still be dominated by a common pattern. A chest X-ray model trained mostly on images from fixed radiography rooms may see portable bedside scans in few updates. If embeddings separate the two acquisition settings, drawing across clusters can expose the model to both more regularly.

`GoldClusterizedBatchSampler` groups samples by embedding similarity, then draws across the resulting pools. The cluster count determines how many pools compete for the available batch positions.

## Mix chest X-ray acquisition settings in each batch

These synthetic vectors stand in for four chest X-rays from fixed radiography rooms (group A) and two portable bedside X-rays (group B). The `mode` field records the groups so we can inspect batch composition. The k-means estimator uses only the vectors. With two clusters, a batch size of two, and `RESTART`, each batch will contain one sample from each group.

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

The epoch draws four batches. Each sample in group B appears twice and each sample in group A appears once. Pass the sampler as `batch_sampler` because it yields a list of dataset positions for each batch.

## Decide what happens when a pool runs out

The three `ExhaustedClusterStrategy` policies make different trade-offs on those same two pools.

![The six example scans under each policy. RESTART draws four mixed batches, STOP draws two, and EXCLUDE draws two mixed batches followed by one fixed-room batch.](figures/batch-exhaustion.png)

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

## Prepare vectors and dataset indices

Use one vector per item and a single clustering fit over the training population so each item belongs to one pool. The clusterizer needs both `idx` and `idx_vector` in the input table. A descriptor configured without a vectorization tool provides `idx` alone, even for flat embeddings. Configure the descriptor's `vectorizer` or pass a separate `GoldVectorizer` to the sampler to produce vector IDs.

Set the clusterizer's `vectorized_key` to the column that contains the vectors. With vectorization inside `GoldDescriptor`, that column is the descriptor's `description_key` (default `"embeddings"`).

The sampler uses `idx` as a dataset position. For a PyTorch `Subset`, supply positions in the base dataset as `idx`. The sampler maps them to subset positions for the loader. [How to inspect an embedding partition](GOLD_CLUSTERIZER.md)

Cluster balancing changes which examples meet in an optimizer update and, with `RESTART`, how often the smaller pools contribute. Embedding-density weights let you adjust a sample's draw frequency or loss contribution based on its local rarity. Compare the policies on the evaluation slices that matter to the model. [Weighting training samples by embedding density](GOLD_DENSITY_WEIGHING.md) · [Zhao and Zhang on stratified minibatch sampling](https://arxiv.org/abs/1405.3080)

Sponsored by [Pixeltable](https://www.pixeltable.com/).
