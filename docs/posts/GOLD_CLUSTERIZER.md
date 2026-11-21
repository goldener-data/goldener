# Use embedding clusters to investigate model errors

## TL;DR

Use `GoldClusterizer` to group similar vectors and retrieve the source samples for review. For monitoring over time, fit a clustering tool once and use `predict` to assign new vectors to the same groups.

## Table of contents

- [Group misrouted support tickets](#group-misrouted-support-tickets)
- [Track the same groups in later sensor windows](#track-the-same-groups-in-later-sensor-windows)
- [How it works](#how-it-works)

An error rate tells you how often a model fails. To decide what to fix, you need to see which errors have something in common. If a classifier sends many support tickets to the wrong queue, grouping their text embeddings can help you find repeated patterns. Read tickets in one cluster to see whether a new form template makes different requests look alike or whether one routing rule needs a clearer definition.

`GoldClusterizer` stores those assignments so you can retrieve members by sample ID. You can use the same workflow before training to find borderline examples and write annotation guidelines.

## Group misrouted support tickets

Eight synthetic vectors stand in for tickets routed to the wrong queue. There are two visibly separate neighborhoods in the toy space. The clusterizer runs one k-means fit over the queue and links each assignment to its original sample.

```python
from uuid import uuid4

import torch
from sklearn.cluster import KMeans

from goldener import GoldClusterizer, GoldSKLearnClusteringTool

vectors = torch.tensor(
    [
        [0.0, 0.0], [0.2, 0.1], [0.1, 0.2], [0.05, 0.15],
        [5.0, 5.0], [5.2, 4.9], [4.8, 5.1], [5.1, 5.2],
    ],
)
sample_ids = [11, 24, 37, 48, 59, 63, 72, 80]
rows = [
    {"idx": sample_id, "idx_vector": pos, "vectorized": vector}
    for pos, (sample_id, vector) in enumerate(zip(sample_ids, vectors))
]

clusterizer = GoldClusterizer(
    table_path=f"review_clusters_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(KMeans(random_state=0)),
)
table = clusterizer.cluster_in_table(rows, n_clusters=2)
groups = {
    cluster_id: clusterizer.get_cluster_indices(
        table, cluster_key="cluster", cluster_idx=cluster_id, idx_key="idx",
    )
    for cluster_id in range(2)
}

assert {frozenset(ids) for ids in groups.values()} == {
    frozenset(sample_ids[:4]), frozenset(sample_ids[4:]),
}
for cluster_id, ids in groups.items():
    print(cluster_id, sorted(ids))
```

The two neighborhoods each yield four IDs. In a real queue, use the IDs to open the original inputs. Review several members and some borderline cases before using a cluster to write an annotation guideline or change a routing rule. The fit produces cluster numbers. The review establishes what the groups mean.

The caller chooses `n_clusters`. A small count gives broader groups to review. A large count can expose narrower patterns and create more groups to inspect. The partition depends on the representation and estimator. If you start from raw data, use `GoldDescriptor` and a vectorization tool, then align the clusterizer's `vectorized_key` with the vector output column. [Lloyd's k-means method](https://doi.org/10.1109/TIT.1982.1056489)

## Track the same groups in later sensor windows

For a sensor-based fault detector, compare the share of incoming windows assigned to a known pattern. These two-dimensional vectors stand in for vibration and temperature features. Fit on a reference period, identify a group from a known high-vibration window, and use `predict` to count its assignments in a later period. Use the same feature extraction and preprocessing in both periods.

```python
import torch
from sklearn.cluster import KMeans

from goldener import GoldSKLearnClusteringTool

reference = torch.tensor([[0.2, 0.1], [0.3, 0.2], [4.0, 0.2], [4.2, 0.3]])
later = torch.tensor([[0.3, 0.1], [3.9, 0.1], [4.0, 0.2], [4.3, 0.3]])
tool = GoldSKLearnClusteringTool(KMeans(random_state=0))

reference_groups = tool.fit(reference, n_clusters=2)
high_vibration_group = reference_groups[-1]
later_groups = tool.predict(later)
reference_count = int((reference_groups == high_vibration_group).sum())
later_count = int((later_groups == high_vibration_group).sum())

assert (reference_count, later_count) == (2, 3)
print(reference_count / len(reference), later_count / len(later))  # 0.5 0.75
```

In this toy comparison, the group's share rises from half to three quarters. Open the corresponding traces to check what changed. Every incoming vector gets an existing cluster ID, so fit a new partition separately if you want to look for patterns the reference groups miss. A fresh fit has its own ID numbering.

## How it works

1. The clusterizer reads each pending vector and its IDs from a dataset or Pixeltable table. `idx_vector` identifies a stored vector assignment, and `idx` links it to the original item. A sensor trace can contribute more than one vector and appear in multiple groups.
2. Each population is fitted independently. If configured, the reducer runs before the fit. `GoldSKLearnClusteringTool` passes the requested cluster count to the estimator and converts the detached 2D tensor to a CPU NumPy array. It returns one integer assignment per row. K-means iterates between assigning each vector to its nearest center and recomputing those centers.
3. The clusterizer writes assignments to Pixeltable. By default, `get_cluster_indices` returns vector IDs. Ask for `idx_key="idx"` to retrieve the source sample IDs. When you call `fit` and `predict` on the tensor tool directly, the estimator in memory supplies the reference partition for later vectors.

Use one fit for a single partition. With `label_key`, Goldener fits each label population separately, which is useful when you want to look for variations within a known class. Setting `chunk` limits memory use by fitting each chunk separately. Both approaches reuse numeric cluster IDs across fits. Query within a label to interpret a label-specific fit. Use an unchunked fit when you need the stored cluster IDs to identify groups in one population, since the table does not store chunk membership. [Reducing the dimensionality of embeddings](GOLD_REDUCTION.md)

Use the Pixeltable output to reopen a completed review. Goldener reuses recorded assignments and fits pending vectors separately on resume. The same numeric ID can therefore refer to different groups in the same table. Use a new table for a fresh fit over the whole population, or call `predict` with the original fitted tool to assign new vectors to existing groups. To control how groups appear during training, give Goldener's batch sampler the training data and a clusterizer configured for one fit. [Building batches from embedding clusters](GOLD_BATCHER.md)
