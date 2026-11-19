# GOLDENER <br> Clustering embeddings to define annotation guidelines

[**Introduction**](#1-introduction) |
[**Components**](#2-components) |
[**Using the groups**](#3-using-the-groups) |
[**Bibliography**](#4-bibliography)

## TLDR

[Goldener](https://github.com/goldener-data/goldener) can group vectorized samples by embedding similarity so you can inspect the modes in a dataset and draft annotation guidelines for each group. `GoldClusterizer` stores one cluster id per vector. `GoldClusterizer.get_cluster_indices(..., idx_key="idx")` returns the original sample IDs in a group.

A cluster is a partition of the vectors that were fitted together. It is not necessarily a ground-truth class, and the integer ids are local to that fit.

The following example runs with Goldener and scikit-learn installed, without downloading a model. The two-dimensional vectors stand in for embeddings already extracted from a pretrained network. Run it in a Python script or notebook; each run uses a fresh local PixelTable table.

```python
from uuid import uuid4

import torch
from sklearn.cluster import KMeans

from goldener import GoldClusterizer, GoldSKLearnClusteringTool

# Two embedding neighborhoods. These integers are sample IDs, not classes.
embeddings = torch.tensor(
    [[0.0, 0.0], [0.2, 0.1], [0.1, 0.2], [0.05, 0.15],
     [5.0, 5.0], [5.2, 4.9], [4.8, 5.1], [5.1, 5.2]],
    dtype=torch.float32,
)
sample_ids = [11, 24, 37, 48, 59, 63, 72, 80]
vectorized = [
    {"idx": idx, "idx_vector": pos, "vectorized": vector}
    for pos, (idx, vector) in enumerate(zip(sample_ids, embeddings))
]

n_clusters = 2
clusterizer = GoldClusterizer(
    table_path=f"cluster_demo_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=n_clusters, random_state=0),
    ),
)
clustered = clusterizer.cluster_in_table(vectorized, n_clusters=n_clusters)

for cluster_id in range(n_clusters):
    sample_indices = GoldClusterizer.get_cluster_indices(
        clustered,
        cluster_key="cluster",
        cluster_idx=cluster_id,
        idx_key="idx",
    )
    print(cluster_id, sorted(sample_indices))
```

This fit uses one population and no chunking, so the two ids are one partition of these eight samples. Label populations and chunking, below, reuse the same integers for separate fits. The toy geometry illustrates the API. It is not a measured improvement in annotation quality.

## 1. Introduction

Datasets often mix several modes: different kinds of images, different kinds of text, or different situations inside one nominal class. Embeddings from a pretrained model place similar samples near each other. Clustering those vectors is a way to list the modes, open a few members of each group, and write annotation guidelines that match what is actually in the data.

The caller chooses how many clusters to request. The algorithm then assigns the fitted vectors to those ids. Samples that share a cluster share a neighborhood in that fit. They can still belong to different labels, and samples that share a label can fall into different clusters. Use the groups to decide what to look at and how to instruct annotators. Keep human labels, or any other supervision, as a separate decision.

`GoldClusterizer` orchestrates that assignment and stores it locally. `GoldSKLearnClusteringTool` adapts a scikit-learn estimator. Training batches that draw from clusters are a later step, described in [Smart batching from embeddings and clustering](GOLD_BATCHER.md).

## 2. Components

### 2.1. Vectorized inputs

Clustering starts from vectors, not from raw images or text. `GoldDescriptor` can extract embeddings, and `GoldVectorizer` can turn those embeddings into vectors. A descriptor that already includes a vectorizer writes them under its `description_key` (`"embeddings"` by default). Point `GoldClusterizer.vectorized_key` at that column. The default `vectorized_key` is `"vectorized"`, which is what the TLDR uses.

`cluster_in_table` and `cluster_in_dataset` accept a PyTorch `Dataset` (a list of dictionaries works) or a PixelTable `Table`. Each dataset item needs a finite floating one-dimensional tensor under `vectorized_key`. A table needs that vector column together with `idx` and `idx_vector`.

`n_clusters` is a required argument of the call and must be greater than 1, even when the wrapped estimator was constructed with a cluster count. `GoldSKLearnClusteringTool` assigns that call value to the estimator's `n_clusters` attribute before fitting. The estimator must expose `n_clusters` and `predict`. The adapter detaches the vectors, moves them to CPU NumPy arrays, and reads `labels_` from `fit`. The cluster table stores those integer ids in `cluster_key` (default `"cluster"`).

```python
dataset_clusterizer = GoldClusterizer(
    table_path=f"cluster_dataset_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=n_clusters, random_state=0),
    ),
)
clustered_dataset = dataset_clusterizer.cluster_in_dataset(
    vectorized, n_clusters=n_clusters,
)
print(type(clustered_dataset).__name__)
```

`cluster_in_table` returns the PixelTable table. `cluster_in_dataset` wraps that table in a `GoldPxtTorchDataset`. `drop_table=True` removes the table only after `cluster_in_dataset` has built the dataset. The default leaves the table in place.

For a dataset input, the vectors are copied into the cluster table and the fit reads them from there. For a table input, the fit reads `vectorized_key` from the source table. Set `include_vectorized_in_table=True` to copy those vectors into the cluster table as well. The default for a table input is to leave them in the source.

`batch_size` and `num_workers` control DataLoader ingestion of a dataset. On a complete run they do not change which vectors are fitted together. `max_batches` stops ingestion early and is intended for tests.

### 2.2. Sample IDs and vector IDs

`idx` identifies the source sample. `idx_vector` identifies one vector and is the primary key of the cluster table. A sample can contribute several vectors, for different parts or views of the same item. Give those rows the same `idx` and a distinct `idx_vector`.

If a dataset item has no `idx_vector`, ingestion assigns one from the DataLoader position (`batch_index * batch_size + offset`). If it also has no `idx`, the sample id copies that generated vector id. Pass explicit ids when the vectors must stay attached to an existing sample key. The generated ids follow the batch size used during ingestion.

`get_cluster_indices` defaults to `idx_key="idx_vector"` and returns a set of distinct ids. Annotation and review need the original samples, so pass `idx_key="idx"`. The same sample id can appear in more than one cluster when its vectors received different ids. `label_key` and `label_value` must be passed together or not at all.

The next example gives sample `11` one vector in each neighborhood. Without `force_same_cluster`, that sample is listed in both clusters.

```python
paired = [
    {"idx": 11, "idx_vector": 0, "vectorized": torch.tensor([0.0, 0.0])},
    {"idx": 11, "idx_vector": 1, "vectorized": torch.tensor([5.0, 5.0])},
    {"idx": 24, "idx_vector": 2, "vectorized": torch.tensor([0.1, 0.1])},
    {"idx": 24, "idx_vector": 3, "vectorized": torch.tensor([0.2, 0.0])},
    {"idx": 37, "idx_vector": 4, "vectorized": torch.tensor([5.1, 4.9])},
    {"idx": 37, "idx_vector": 5, "vectorized": torch.tensor([4.9, 5.2])},
]
per_vector = GoldClusterizer(
    table_path=f"cluster_per_vector_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=2, random_state=0),
    ),
)
per_vector_table = per_vector.cluster_in_table(paired, n_clusters=2)
for cluster_id in range(2):
    print(
        cluster_id,
        sorted(GoldClusterizer.get_cluster_indices(
            per_vector_table,
            cluster_key="cluster",
            cluster_idx=cluster_id,
            idx_key="idx",
        )),
    )
```

`force_same_cluster=True` writes the cluster id by sample `idx`. Every stored row with that sample id receives the id, including rows that were not in the current fitted chunk. If the sample's vectors are assigned different ids in one fit, a later write for that sample replaces the earlier one. The option copies one assignment onto the sample. It does not fit a single combined representation of the sample, and a later chunk that contains the same sample can replace the stored id again.

```python
forced = GoldClusterizer(
    table_path=f"cluster_forced_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=2, random_state=0),
    ),
    force_same_cluster=True,
)
forced_table = forced.cluster_in_table(paired, n_clusters=2)
memberships = [
    GoldClusterizer.get_cluster_indices(
        forced_table,
        cluster_key="cluster",
        cluster_idx=cluster_id,
        idx_key="idx",
    )
    for cluster_id in range(2)
]
print([sorted(members) for members in memberships])
```

With this flag, each of samples `11`, `24`, and `37` appears in only one of the returned sets. Which id sample `11` keeps is the id written last for that sample during the fit.

### 2.3. One fit per label

`label_key=None` (the default) fits every pending vector as one population. Set `label_key` to a column of string labels to fit each distinct label on its own. Every label is asked for the same `n_clusters`. Cluster `0` inside one label is a separate fit from cluster `0` inside another label. The shared integers do not mean a shared mode.

Retrieval filters one label only when both `label_key` and `label_value` are set. The cluster table stores `label_key` as a string column.

```python
labeled = [
    {**row, "label": "reports" if pos < 4 else "photos"}
    for pos, row in enumerate(vectorized)
]
by_label = GoldClusterizer(
    table_path=f"cluster_labels_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=2, random_state=0),
    ),
    label_key="label",
)
labeled_table = by_label.cluster_in_table(labeled, n_clusters=2)
reports = GoldClusterizer.get_cluster_indices(
    labeled_table,
    cluster_key="cluster",
    cluster_idx=0,
    label_key="label",
    label_value="reports",
    idx_key="idx",
)
print(sorted(reports))
```

Each label in this example has four vectors, and each label is partitioned into `n_clusters` groups. Inspect a label with its own `label_value`. A loop over cluster ids without that filter mixes the separate fits.

### 2.4. An optional reducer

Pass a `GoldReductionTool` as `reducer` to transform vectors before the clustering fit. A reducer that implements `fit` (`GoldReductionToolWithFit`, including `GoldSKLearnReductionTool`) is `fit_transform`'d on each fitted population. A reducer with only `transform` applies that transform and is not refit. The clustering adapter then sees the reduced rows, in the same order.

The scope of a fittable reducer is that population: one chunk of one label, or the whole pending population when there is no chunking and no label key. A later chunk fits the reducer again. Reduced coordinates from different chunks are not one shared projection of the dataset.

```python
from sklearn.decomposition import PCA

from goldener import GoldSKLearnReductionTool

wide = torch.randn(12, 8, generator=torch.Generator().manual_seed(0))
wide_rows = [
    {"idx": pos, "idx_vector": pos, "vectorized": vector}
    for pos, vector in enumerate(wide)
]
reduced = GoldClusterizer(
    table_path=f"cluster_reduced_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=2, random_state=0),
    ),
    reducer=GoldSKLearnReductionTool(PCA(n_components=2)),
)
print(reduced.cluster_in_table(wide_rows, n_clusters=2).count())
```

The scikit-learn reduction adapter moves inputs to CPU for NumPy and restores the input device and dtype on the transformed tensor. The clustering adapter does not use that restoration contract: it consumes the reduced tensor and stores integer labels.

### 2.5. Chunking

`chunk=None` (the default) fits each pending population in one call. Set `chunk` to a positive integer to split the pending vector ids into random groups of at most that size. `GoldClusterizer.random_state` controls only that assignment. It does not seed the wrapped estimator. Set `random_state` on the estimator itself, as the KMeans examples do, when the estimator's own initialization should be repeatable.

Each chunk is fitted separately with the same `n_clusters`. The table stores those ids side by side. Cluster `0` from one chunk and cluster `0` from another chunk are not one globally fitted partition. Do not present a chunked `cluster` column, or a loop over `range(n_clusters)` on that column, as a single clustering of the whole dataset.

Every fitted chunk has to be large enough for the estimator. KMeans needs at least `n_clusters` vectors in each fit. For `n` pending vectors, the assignment makes `ceil(n / chunk)` fits of nearly equal size, and the smallest fit has `n // ceil(n / chunk)` vectors. That size can be well below `chunk`: 9 vectors with `chunk=4` become three fits of 3, which is too small for `n_clusters=4`. Choose `chunk` so the smallest fit has at least `n_clusters` vectors. With a `label_key`, use each label's pending vector count as `n`. When `chunk` is `None` or at least the population size, that population stays in one fit.

```python
chunked = GoldClusterizer(
    table_path=f"cluster_chunks_{uuid4().hex}",
    clustering_tool=GoldSKLearnClusteringTool(
        KMeans(n_clusters=2, random_state=0),
    ),
    chunk=4,
    random_state=0,
)
chunked_table = chunked.cluster_in_table(vectorized, n_clusters=2)
for cluster_id in range(2):
    print(
        cluster_id,
        sorted(GoldClusterizer.get_cluster_indices(
            chunked_table,
            cluster_key="cluster",
            cluster_idx=cluster_id,
            idx_key="idx",
        )),
    )
```

Eight vectors and `chunk=4` produce two fits. The printed sets group rows that happen to share an integer. They are not the two neighborhoods from the single fit in the TLDR.

### 2.6. Local results and restart

Results are stored in a local PixelTable table at `table_path`. The primary key is `idx_vector`. Rows are inserted with a null cluster, and the fit fills that column. `allow_existing=True` (the default) reopens an existing table at that path. `allow_existing=False` raises `ValueError` when the table already exists. The existing table must use `idx_vector` as its primary key.

A finished table is reused. Calling `cluster_in_table` again does not insert duplicate rows and does not refit vectors whose cluster id is already stored.

```python
replayed = clusterizer.cluster_in_table(vectorized, n_clusters=n_clusters)
print(replayed.count(), clustered.count())
```

Restart applies only to rows that still have a null cluster. Those pending vector ids are chunked and fitted on their own. Completed rows stay as they were and are excluded from the new fit. Chunk membership is not stored. After an interrupted run, the remaining vectors can therefore receive different ids from the ids they would have received in one uninterrupted fit, and a fittable reducer is fit only on the pending population. Rechunking does not rebuild a single partition, and this post does not claim that a resumed run matches a fresh run.

Changing `n_clusters`, the clustering tool, the reducer, or `chunk` between calls leaves completed rows untouched. Delete or replace the table to recompute them.

Distributed clustering is currently unsupported. `distribute=True` raises `NotImplementedError` at construction and on the `distribute` setter.

## 3. Using the groups

For a single fit, the TLDR loop is the inspection step: take each cluster id, retrieve sample ids with `idx_key="idx"`, and open those samples in the original dataset. The set has no order. Sample a few members and write the guideline from the content you see, then review the boundary cases rather than treating the integer as a class name.

When labels or chunks were used, restrict the review to one fitted population before treating an id as a group. Per-label retrieval needs `label_key` and `label_value`. A chunked column needs a single unchunked fit, or an external record of chunk membership, before its ids can be read as one partition. Goldener does not store that chunk membership today.

Those sample ids are also the link into later steps. [Smart batching from embeddings and clustering](GOLD_BATCHER.md) uses a `GoldClusterizer` to balance training batches. The batch sampler consumes the stored ids; it does not turn them into ground-truth labels.

## 4. Bibliography

Related reading on clustering as a way to group observations. These references describe the general methods. They are not benchmark evidence for annotation quality or for Goldener's orchestration.

- Lloyd, S. [Least squares quantization in PCM](https://doi.org/10.1109/TIT.1982.1056489). IEEE Transactions on Information Theory, 1982. The k-means objective used by the `KMeans` estimator in the examples.
- Jain, A. K. [Data clustering: 50 years beyond K-means](https://doi.org/10.1016/j.patrec.2009.09.011). Pattern Recognition Letters, 2010. Clustering as exploratory grouping, with cluster count and representation chosen by the analyst.
