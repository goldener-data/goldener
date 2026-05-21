# GOLDENER <br> Smart batching from embeddings and clustering

[**Introduction**](#1-introduction) |
[**Components**](#2-components) |
[**Smart batching**](#3-smart-batching) |
[**Bibliography**](#4-bibliography) |
[**Authors**](#5-authors)|
[**Miscellaneous**](#6-miscellaneous)

## TLDR

In [Goldener](https://github.com/goldener-data/goldener), pretrained models and clustering algorithms can be leveraged to perform smart batching (as opposed to the usual random shuffling).

```python
# Create a descriptor to access the semantics representation
embedder_config = GoldTorchEmbeddingToolConfig(
    model=my_model,
    layers=my_layers,
    layer_fusion=EmbeddingFusionStrategy.AVERAGE,
    group_fusion=EmbeddingFusionStrategy.CONCAT,
    channel_pos=1,
)
embedder = GoldTorchEmbeddingTool(embedder_config)
vectorizer = TensorVectorizer(
    keep=Filter2DWithCount(),
    fusion_strategy=EmbeddingFusionStrategy.AVERAGE,
    transform_y=None,
    channel_pos=1
)
gold_descriptor = GoldDescriptor(
    table_path="my_table_for_description",
    embedder=embedder,
    vectorizer=vectorizer,
)


# Create a clusterizer to organize the samples
gold_clusterizer = GoldClusterizer(
    table_path="my_table_for_clustering",
    clustering_tool=GoldSKLearnClusteringTool(
        tool=KMeans(
            n_clusters=n_clusters,
            random_state=42,
        )
    ),
)

# Create a semantics aware batch sampler
batch_sampler = GoldClusterizedBatchSampler(
    dataset=my_dataset,
    descriptor=gold_descriptor,
    vectorizer=None,
    batch_size=batch_size,
    n_clusters=n_clusters,
    clusterizer=gold_clusterizer,
    force_same_size=False,
    shuffle=True,
    generator=generator,
    strategy=ExhaustedClusterStrategy.EXCLUDE,
)

# Iterate over the dataset
dataloader = DataLoader(
    dataset=my_dataset,
    batch_sampler=batch_sampler,
)
for batch in dataloader:
  # do something with the batch
```


## 1. Introduction

[Goldener](https://github.com/goldener-data/goldener) is an open source Python library focusing on data orchestration for efficient Machine Learning (ML). It provides features to sample, split, organize, annotate, and curate data based on embeddings/features in order to make the full ML lifecycle more efficient, from training to monitoring.

Deep learning models are trained by iterating multiple times over the different samples of the dataset. Each iteration takes a mini-batch with a specific number of elements. The standard practice for creating mini-batches is random sampling from the dataset without replacement. Once the dataset is exhausted, the drawing is restarted and new mini-batches are sampled. Every mini-batch is then composed of its own data distribution.

The distribution of each mini-batch sequentially guides the model toward a local minimum, and this minimum can shift depending on the random selection of elements. Indeed, each mini-batch is leveraged to update the weights of the model and the weights update depends on the loss value. When a mini-batch exhibits a distribution drastically different from the others, it can push the model weights in the wrong direction — causing training instability and degraded performance.


In this post, we introduce how [Goldener](https://github.com/goldener-data/goldener) enables smarter mini-batches by leveraging pretrained networks and clustering algorithms. This semantics aware batching aims to provide mini-batches with more stable distributions, hence ensuring training stability and better model convergence.


## 2. Components

### 2.1. Semantics extraction

[Goldener](https://github.com/goldener-data/goldener) is designed to process datasets of any size. To accommodate this, Goldener utilizes a local internal storage managed by [Pixeltable](https://www.pixeltable.com/). With this open source Python library, Goldener locally saves its output but also the different objects required to run/restart a task.

In Goldener, the smart sampling is based on the semantics description of the data provided by the embeddings computed from pretrained networks. To remain data- and task-agnostic, the selection and sampling tools in [Goldener](https://github.com/goldener-data/goldener) are all designed to process vectors. Thus, Goldener provides tools to describe data semantics, vectorize this information, and, when necessary, filter for targeted elements.

More details about the different building blocks are available in this [article](https://huggingface.co/blog/Yann-CV/goldener-smart-split#2-components).

```python
embedder_config = GoldTorchEmbeddingToolConfig(
    model=my_model,
    layers=my_layers,
    layer_fusion=EmbeddingFusionStrategy.AVERAGE,
    group_fusion=EmbeddingFusionStrategy.CONCAT,
    channel_pos=1,
)
embedder = GoldTorchEmbeddingTool(embedder_config)
vectorizer = TensorVectorizer(
    keep=Filter2DWithCount(),
    fusion_strategy=EmbeddingFusionStrategy.AVERAGE,
    transform_y=None,
    channel_pos=1
)
gold_descriptor = GoldDescriptor(
    table_path="my_table_for_description",
    embedder=embedder,
    vectorizer=vectorizer,
    data_key="data",
    target_key="target",
    label_key="label",
    description_key="embeddings",
)
description_table = gold_descriptor.describe_in_table(my_dataset)
```


### 2.2. Data clustering

The main entry point in the code is [GoldClusterizer](https://github.com/goldener-data/goldener/blob/main/goldener/clusterize.py).

Multiple algorithms exist to cluster data based on the `Description` (vectors) of the samples. To remain provider-agnostic, [Goldener](https://github.com/goldener-data/goldener) requires them to be encapsulated within a GoldClusteringTool class. Currently, Goldener natively provides GoldSKLearnClusteringTool, which grants easy access to any Scikit-Learn clustering algorithm.

In [Goldener](https://github.com/goldener-data/goldener), the `GoldClusterizer` class orchestrates the clustering of samples from a `GoldClusteringTool`. In `GoldClusterizer`, the internal `Table` storing the clustering result is created and partly populated (all columns except the cluster information one) before applying any clustering algorithm. Then, the clustering process is stratified between the classes when the `label_key` is provided. When all the vectors cannot be processed at once, the user can specify a chunk size to split the clustering process in multiple smaller chunks of data. The assignment of the vectors to a given chunk is done randomly. Additionally, when the memory constraints are very restrictive, `GoldClusterizer` can integrate a `GoldReductionTool` in order to reduce the dimensions of the vectors. The available reduction tools are UMAP, PCA, t-SNE, and GaussianRandomProjection. An interface to use any PyTorch module as `GoldReductionTool` is also available.

Depending on the dataset and `Description`, multiple vectors can come from the same sample. Thus, in [Goldener](https://github.com/goldener-data/goldener), the clustering algorithm might be applied multiple times on the same samples, and this single sample may be assigned to multiple clusters. The target number of clusters is specified during the function call. The indices of the vectors are used to populate the cluster column with the cluster index returned by the clustering algorithm.

In [Goldener](https://github.com/goldener-data/goldener), `GoldClusterizer` is callable from either a PyTorch `Dataset` or a Pixeltable `Table`. If a `Dataset` is provided, each sample is expected to be accessible as a dictionary containing the keys defined in `vectorized_key`, and `label_key` attributes (if it is not initially in this format, a custom collate function must be provided). During the initialization of the clustering `Table`, the input data is processed sequentially batch by batch using a PyTorch `DataLoader`. Then, the data is clustered from the internal Pixeltable `Table`, label by label if the `label_key` is provided, and chunk by chunk if required. Finally, the `GoldClusterizer` returns a PyTorch `Dataset` via `cluster_in_dataset` or a Pixeltable `Table` via `cluster_in_table`. At the end, the clustering indices are stored within the column defined by the `clusterized_key` attribute. The output also includes the `idx` and `idx_vector` fields, with `idx` storing the index of the sample and `idx_vector` the index of the vector (each sample might be described by multiple vectors).

```python
# Create a clusterizer to organize the samples
gold_clusterizer = GoldClusterizer(
    table_path="my_table_for_clustering",
    clustering_tool=GoldSKLearnClusteringTool(
        tool=KMeans(
            n_clusters=n_clusters,
            random_state=42,
        )
    ),
)
clusterized = gold_clusterizer.cluster_in_dataset(
    my_dataset, n_clusters=10
)
```

## 3. Smart batching

The main entry point in the code is [GoldClusterizedBatchSampler](https://github.com/goldener-data/goldener/blob/main/goldener/organize.py)

In [Goldener](https://github.com/goldener-data/goldener), In Goldener, smart batching is implemented as a custom batch sampler (sampler returning the list of index per batch) based on:
* Vectors characterizing the samples/elements from both local and global semantics.
* A clustering algorithm relying on vectors to gather together samples sharing the same distribution of features.

Unlike traditional random shuffling, smart batching optimizes batch composition to maintain a steady, representative feature distribution across batches.

In [Goldener](https://github.com/goldener-data/goldener), the vectors characterizing the data can be obtained from a `GoldDescriptor` object and the clustering of the samples from a `GoldClusterizer` class. The `GoldClusterizedBatchSampler` class leverages these two objects during its initialization to access the cluster index of all samples. Like the usual batch sampler from Pytorch, the `__iter__` method is returning a list of size `batch_size` corresponding to the next sample to draw. For each iteration, a new batch of indices is drawn to ensure samples are distributed across different clusters. If the number of clusters is greater than the batch size, some clusters are selected randomly. When it is lower, the random sampling is done multiple times among the clusters.

Depending on the clustering algorithm, the size of the clusters can be different. The `strategy` argument handles this variance with three distinct policies for exhausted clusters (i.e., those whose samples have been fully consumed):
- The cluster is reset, meaning its samples can be drawn again before the iterator exhaustion (all samples have been consumed at least once).
- The cluster is excluded, forcing subsequent iterations to sample exclusively from the non exhausted clusters until the iterator exhaustion.
- The iterator stops early as exhausted, meaning some samples in larger clusters may not be sampled during the epoch.

As the PyTorch batch sampler, `GoldClusterizedBatchSampler` can be provided to a Pytorch DataLoader in order to iterate over the dataset during the training of the model.

```python
gold_descriptor = GoldDescriptor(...) # descriptor with vectorizer included
gold_clusterizer = GoldClusterizer(...)

batch_sampler = GoldClusterizedBatchSampler(
    dataset=my_dataset,
    descriptor=gold_descriptor,
    vectorizer=None,
    batch_size=batch_size,
    n_clusters=n_clusters,
    clusterizer=gold_clusterizer,
    force_same_size=False,
    shuffle=True,
    generator=generator,
    strategy=ExhaustedClusterStrategy.EXCLUDE,
)

# Iterate over the dataset
dataloader = DataLoader(
    dataset=my_dataset,
    batch_sampler=batch_sampler,
)
for batch in dataloader:
  # do something with the batch
```


## 4. Bibliography

- Hacohen, G., et al. On the power of curriculum learning in training deep networks. International conference on machine learning. PMLR, 2019.
- Zhao, P., et al. Accelerating minibatch stochastic gradient descent using stratified sampling. arXiv preprint arXiv:1405.3080. 2014.

## 5. Authors

[Yann Chéné, PhD,](https://huggingface.co/Yann-CV) is a Machine Learning (ML) engineer currently working at [Scortex](https://scortex.io/) - a company leveraging computer vision to automate manufacturing quality control. Within Scortex, he is involved in tasks from research to product integration and MLOps. His current focus is on improving the state of the art in image anomaly detection. Yann is also the creator of [Goldener](https://pypi.org/project/goldener/), an open source Python data orchestrator. Goldener proposes features to sample, split, organize, annotate, and curate data based on model embeddings/features in order to make the full ML lifecycle more efficient.


## 6. Miscellaneous

Sponsored by [Pixeltable](https://www.pixeltable.com/): Multimodal Data, Made Simple. Video, audio, images, and documents as first-class data types, with storage, orchestration, and retrieval unified under one table interface.
