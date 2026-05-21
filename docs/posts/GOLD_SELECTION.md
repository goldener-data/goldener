# GOLDENER <br> Smart sampling from embeddings and coreset selection

[**Introduction**](#1-introduction) |
[**Semantics extraction**](#2-semantics-extraction) |
[**Smart split**](#3-smart-sampling) |
[**Bibliography**](#4-bibliography) |
[**Authors**](#5-authors)|
[**Miscellaneous**](#6-miscellaneous)

## TLDR

In [Goldener](https://github.com/goldener-data/goldener), pretrained models and coreset selection algorithms can be leveraged to perform smart sampling (as opposed to the usual random sampling).

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
    data_key="data",
    target_key="target",
    label_key="label",
    description_key="embeddings",
)
description_table = gold_descriptor.describe_in_table(dataset)

# Create a selector to select the samples
selection_tool = GoldGreedyKCenterSelectionTool(
    device=torch.device("cuda"),
    distance=DistanceType.EUCLIDEAN
)
gold_selector = GoldSelector(
    table_path="my_table_for_selection",
    selection_tool=selection_tool,
    selection_key="selected",
    label_key="label",
    vectorized_key="vectorized",
)

selection_table = gold_selector.select_in_table(
    description_table, select_size=100, value="training"
)
selected_indices = gold_selector.get_selection_indices(selection_table, "training", "selected")
```


## 1. Introduction

[Goldener](https://github.com/goldener-data/goldener) is an open source Python library focusing on data orchestration for efficient Machine Learning (ML).
It provides features to sample, split, organize, annotate, and curate data based on embeddings/features in order to make the full ML lifecycle more efficient, from training to monitoring.

In the ML lifecycle, data sampling can be done at different steps:
- Among historical unannotated data to trigger the initial training or retraining of the model.
- Among training data (annotated or not) to curate and reduce the size of the training set.
- Among inference data to monitor model performance.

In all cases, a random selection process can skew both actual performance (suboptimal model training) and the perception of that performance (over- or under-confidence in the model). In this post, we introduce how [Goldener](https://github.com/goldener-data/goldener) gives access to smarter data sampling by leveraging pretrained networks and coreset selection algorithms. This semantics aware sampling aims to optimize both the capacity and the understanding of the real behavior of the deployed model.


## 2. Semantics extraction

[Goldener](https://github.com/goldener-data/goldener) is designed to process datasets of any size. To accommodate this, Goldener utilizes a local internal storage managed by [Pixeltable](https://www.pixeltable.com/). With this open source Python library, Goldener locally saves its output but also the different objects required to run/restart a task.

In Goldener, the smart sampling is based on the semantics description of the data provided by the embeddings computed from pretrained networks. In order to be data and task agnostic, the selection/sampling tools in [Goldener](https://github.com/goldener-data/goldener) are all designed to process vectors. Thus, Goldener provides tools to describe data semantics, vectorize this information, and, when necessary, filter for targeted elements.

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
    kkeep=Filter2DWithCount(),
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
description_table = gold_descriptor.describe_in_table(dataset)
```

## 3. Smart sampling

The main entry point in the code is [GoldSelector](https://github.com/goldener-data/goldener/blob/main/goldener/select.py).

Once the semantics `Description` (vectors) of the dataset is available, smart sampling means leveraging the distribution of the vectors in the semantics space to select specific samples. This automated distribution analysis is then performed using coreset subsampling algorithms applied directly to the data pool. Depending on the algorithm, the selection can approximate the global distribution, ensure all data clusters are represented, or even combine both strategies.

In [Goldener](https://github.com/goldener-data/goldener), the coreset subsampling algorithm must be implemented within a class inheriting from the `GoldSelectionTool` class. Goldener natively supports several strategies:

* `GoldGreedyKernelPointsSelectionTool` based on the `GreedyKernelPoints` class from [Coreax](https://coreax.readthedocs.io/en/latest/index.html)
* `GoldGreedyKCenterSelectionTool` based on an internal reimplementation of the Greedy K-Center algorithm
* `GoldZCoreSelectionTool` based on an internal reimplementation of the ZCore method
`GoldSelectionTool` assumes that all the data to select from fits in memory (RAM or VRAM). Those selection tools take the vectorized data as input and return a list of integers specifying which vectors have been selected.

In [Goldener](https://github.com/goldener-data/goldener), the `GoldSelector` class orchestrates the selection of samples from a `GoldSelectionTool`. In `GoldSelector`, the internal selection `Table` is created and partly populated (all columns except the selection one) before applying any selection algorithm. Then, the selection process is stratified between the classes when the `label_key` is provided. When all the vectors cannot be processed at once, the user can specify a chunk size to split the selection process in multiple smaller chunks of data. Vectors are assigned to chunks randomly. Additionally, under strict memory constraints, `GoldSelector` can integrate a `GoldReductionTool` in order to reduce the dimensions of the vectors. The available reduction tools are UMAP, PCA, t-SNE, and GaussianRandomProjection. An interface to use any PyTorch module as `GoldReductionTool` is also available.

Depending on the dataset and `Description`, multiple vectors can come from the same sample. Thus, in [Goldener](https://github.com/goldener-data/goldener), the selection algorithm might be applied multiple times before reaching the requested number of samples. This selection target can be specified as a fixed integer value or a float corresponding to a ratio of the full dataset. The indices of the selected vectors are used to populate the selection column with the specified value. Once a vector of a sample is selected, its other vectors are removed from the next selection round.

In [Goldener](https://github.com/goldener-data/goldener), `GoldSelector` is callable from either a PyTorch `Dataset` or a Pixeltable `Table`. If a `Dataset` is provided, each sample is expected to be accessible as a dictionary containing the keys defined in `vectorized_key`, `label_key`, `idx`, and `idx_vector` attributes (if it is not initially in this format, a custom collate function must be provided). During the initialization of the selection `Table`, the input data is processed sequentially, batch by batch, using a PyTorch `DataLoader`. Then, the data is selected from the internal Pixeltable `Table`, label by label if the `label_key` is provided, and chunk by chunk if required. Finally, the `GoldSelector` returns a PyTorch `Dataset` via `select_in_dataset` or a Pixeltable `Table` via `select_in_table`. At the end, the selection status is stored within the column defined by the `selection_key` attribute. The output also includes the `idx` and `idx_vector` fields, with `idx` storing the index of the sample and `idx_vector` the index of the vector (each sample might be described by multiple vectors).

```python
# Create a selection tool to select the samples
selection_tool = GoldGreedyKCenterSelectionTool(
    device=torch.device("cuda"),
    distance=DistanceType.EUCLIDEAN
)

# Select the samples and store the result
gold_selector = GoldSelector(
    table_path="my_table_for_selection",
    selection_tool=selection_tool,
    selection_key="selected",
    label_key="label",
    vectorized_key="vectorized",
)
selection_table = gold_selector.select_in_table(
    description_table, select_size=100, value="training"
)
selected_indices = gold_selector.get_selection_indices(selection_table, "training", "selected")
```


## 4. Bibliography

The readers of this post might be interested in the following resources:

* Moser, Brian B., et al. A coreset selection of coreset selection literature: Introduction and recent advances. arXiv preprint arXiv:2505.17799. 2025.
* Griffin, Brent A., et al. Zero-Shot Coreset Selection via Iterative Subspace Sampling. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2026.

## 5. Authors

[Yann Chéné, PhD,](https://huggingface.co/Yann-CV) is a Machine Learning (ML) engineer currently working at [Scortex](https://scortex.io/) - a company leveraging computer vision to automate manufacturing quality control. Within Scortex, he is involved in tasks from research to product integration and MLOps. His current focus is on improving the state of the art in image anomaly detection. Yann is also the creator of [Goldener](https://pypi.org/project/goldener/), an open source Python data orchestrator. Goldener proposes features to sample, split, organize, annotate, and curate data based on model embeddings/features in order to make the full ML lifecycle more efficient.


## 6. Miscellaneous

Sponsored by [Pixeltable](https://www.pixeltable.com/): Multimodal Data, Made Simple. Video, audio, images, and documents as first-class data types, with storage, orchestration, and retrieval unified under one table interface.
