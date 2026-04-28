# GOLDENER <br> Smart data split from embeddings and coreset selection

[**Introduction**](#1-introduction) |
[**Components**](#2-components) |
[**Smart split**](#3-smart-split) |
[**Bibliography**](#4-bibliography) |
[**Authors**](#5-authors)|
[**Miscellaneous**](#6-miscellaneous)

## TLDR

In [Goldener](https://github.com/goldener-data/goldener), pretrained models and coreset selection algorithms can be leveraged to make smart splits (as opposed to the usual random split).

```python
# Create an embedder to access the semantic representation
embedder_config = GoldTorchEmbeddingToolConfig(
  model=my_model,
  layers=my_layers,
  layer_fusion=EmbeddingFusionStrategy.AVERAGE,
  group_fusion=EmbeddingFusionStrategy.CONCAT,
  channel_pos=1,
)
embedder = GoldTorchEmbeddingTool(embedder_config)
vectorizer = TensorVectorizer(
  keep=keep,
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

# Create a selector to select the samples
selection_tool = GoldGreedyKCenterSelectionTool(
  distance=DistanceType.EUCLIDEAN
)
gold_selector = GoldSelector(
  table_path="my_table_for_selection",
  selection_tool=selection_tool,
  selection_key="selected",
  label_key="label",
  vectorized_key="vectorized",
)

# Create a splitter to orchestrate the split
gold_splitter = GoldSplitter(
    sets=[GoldSet("train", 0.7), GoldSet("val", 0.3)],
    descriptor=gold_descriptor,
    selector=gold_selector,
)

# Split the data from its semantic representation
split_table = gold_splitter.split_in_table(dataset)
splits = gold_splitter.get_split_indices(
    split_table, selection_key="selected", idx_key="idx"
)
train_indices = splits["train"]
val_indices = splits["val"]
```


## 1. Introduction

[Goldener](https://github.com/goldener-data/goldener) is an open source Python library focusing on data orchestration for efficient Machine Learning (ML).
It provides features to sample, split, organize, annotate, and curate data based on embeddings/features in order to make the full ML lifecycle more efficient, from training to monitoring.

In this post, we introduce how [Goldener](https://github.com/goldener-data/goldener) gives access to smarter data splits by leveraging pretrained networks and coreset selection algorithms. We start by detailing the different components of Goldener involved in its smart data split feature. Then, we describe the internal process of a `GoldSplitter`, the Goldener class responsible for orchestrating the split of data among different sets.

## 2. Components

### 2.1 Internal data storage

[Goldener](https://github.com/goldener-data/goldener) is designed to process datasets of any size:
* For most datasets and tasks, it is likely not possible to keep all the data in RAM/VRAM memory at the same time.
* Machine preemption might happen during a task. It is likely to stop the task and then come back to finish it.
* Multiple steps in the ML lifecycle might require leveraging embeddings/features, it is likely to reuse the same semantic representation multiple times.
These assumptions enforce the usage of an internal data storage in Goldener. To manage this internal storage, Goldener uses [Pixeltable](https://www.pixeltable.com/). With this open source Python library, Goldener locally saves its output but also the different objects required to run/restart a task.

With Pixeltable, all the objects are stored in a `Table` in which the columns can have different types, from simple text to tensors, dictionaries of variables, and links to cloud storage. In this `Table`, Pixeltable is enforcing new entry types and keeping track of data lineage information. It is also possible to convert the `Table` to a PyTorch `Dataset` and to store the created `Table` in a dedicated cloud storage. The ability to save different inputs, outputs, and intermediate objects makes [Goldener](https://github.com/goldener-data/goldener) fully task agnostic, with the ability to process text, image, video, or any other data type.

In most of the [Goldener](https://github.com/goldener-data/goldener) features/tools (all those assuming the full data loading is not possible), the user is expected to specify the path for an internal [Pixeltable](https://www.pixeltable.com/) `Table`. This `Table` is then used to store the intermediate information and outputs. During the object initialization or feature call, the `Table` is either created or reloaded if it already exists. When it exists, the expected columns and their types are validated against the new task and provided data. Finally, Goldener detects the status of the task and then starts it from scratch or restarts it where it was stopped.

When a [Goldener](https://github.com/goldener-data/goldener) tool requires iterating over the data, either it is loaded sequentially from a PyTorch DataLoader or it is loaded directly from the internal [Pixeltable](https://www.pixeltable.com/) `Table`. Finally, all the main features are available through the creation of specific `GoldDoer` classes. All `GoldDoer` classes expose two main public methods. The first one returns the output as a Pixeltable `Table` and follows the `do_something_in_table` name convention. The second one returns a dataset and follows the `do_something_in_dataset` name convention. Behind the scenes, the second one is calling the first one and it is possible to keep or not the associated internal Pixeltable `Table`. If the output is finally a Dataset, all its content is stored from the internal Pixeltable `Table` in Parquet format.

### 2.2. Description with pretrained models

The main entry point in the code is [GoldDescriptor](https://github.com/goldener-data/goldener/blob/main/goldener/describe.py).

In [Goldener](https://github.com/goldener-data/goldener), the smart split selects the samples for the sets based on vectors/features/embeddings representing the samples. Depending on the task, the raw vectors characterizing the samples can be used directly by the selection algorithm. However, for more complex data (sentences, images, temporal sequences, among others), the representation of every sample/element must be based on both local (unique value/vector and its direct neighborhood) and global (relative comparison with other values/vectors from the same sample) contexts.

The representation aggregating the local and global semantic of a sample/element, called `Description` in [Goldener](https://github.com/goldener-data/goldener), can be the embeddings/features extracted from a pretrained model. In these pretrained models, the information flow is processed through successive blocks/layers moving the input data into a space allowing the model to succeed in a downstream task. Thus, the different blocks/layers give access to different semantic levels of this input data.

In [Goldener](https://github.com/goldener-data/goldener), the `Description` can be extracted from `GoldEmbeddingTool` objects. For instance, the `GoldTorchEmbeddingTool` is available to use any [PyTorch](https://pytorch.org/) model. In `GoldTorchEmbeddingTool`, the model is associated with a list of layer names (named modules in the model) or a dictionary of layer name lists in order to define the layer/blocks used to describe the samples. The dictionary format enables a first aggregation of the layers within the same group before aggregating all groups. Goldener offers different strategies to merge the layers/groups as a single `Description`: concatenation, addition, average, or maximum.

The [Goldener](https://github.com/goldener-data/goldener) class orchestrating the description of a full dataset is `GoldDescriptor`. It is callable from either a PyTorch `Dataset` or a Pixeltable `Table`. If a `Dataset` is provided, all samples are expected to be accessible as a dictionary containing the keys defined in `data_key`, `target_key`, and `label_key` attributes (if it is not initially in this format, a custom collate function must be provided). The input data is processed sequentially batch by batch using a PyTorch `DataLoader`. At the end of its task, the `GoldDescriptor` returns a PyTorch `Dataset` via `describe_in_dataset` or a Pixeltable `Table` via `describe_in_table`. At the end, the description is stored within the column defined by the `description_key` attribute. The output also includes the `idx` field storing the index of the sample.

```python
# Create an embedder to access the semantic representation
embedder_config = GoldTorchEmbeddingToolConfig(
  model=my_model,
  layers=my_layers,
  layer_fusion=EmbeddingFusionStrategy.AVERAGE,
  group_fusion=EmbeddingFusionStrategy.CONCAT,
  channel_pos=1,
)
embedder = GoldTorchEmbeddingTool(embedder_config)

# Extract and store the semantic representation
gold_descriptor = GoldDescriptor(
  table_path="my_table_for_description",
  embedder=embedder,
  data_key="data",
  target_key="target",
  label_key="label",
  description_key="embeddings",
)
description_table = gold_descriptor.describe_in_table(my_dataset)
```

### 2.3. Vectorization of description

The main entry point in the code is [GoldVectorizer](https://github.com/goldener-data/goldener/blob/main/goldener/vectorize.py).

Depending on the input data and the model, the `Description` of each sample might be a multidimensions tensor instead of a single vector (a feature map from an image, temporal evolution description from a signal, among others). In [Goldener](https://github.com/goldener-data/goldener), the selection tools are designed to process vectors, this makes them input type/task agnostic. Thus, Goldener proposes a `GoldVectorizer` in order to move from a batch of tensors to a list of vectors.

The [Goldener](https://github.com/goldener-data/goldener) orchestrating the vectorization of a full dataset is `GoldVectorizer`.
Its main building block is the `TensorVectorizer` which vectorizes its input regardless of the tensor shape. Within the input tensor, some vectors can be selected or ignored from the `keep`, `remove`, `random` attributes. Furthermore, when more than one vector is still present in the description of a sample, the remaining vectors can be aggregated together (same possibilities as for the layers). Finally, depending on the task and dataset, some local annotations might be available. Then, the vectorizer can use this information to restrict the focus of the description on the specific annotated elements. If the annotation is required to be adapted to the description scale, the user can specify a specific transform in order to adapt the annotation.

In [Goldener](https://github.com/goldener-data/goldener), `GoldVectorizer` is callable from either a PyTorch `Dataset` or Pixeltable `Table`. If a `Dataset` is provided, each sample is expected to be accessible as a dictionary containing the keys defined in `data_key`, `target_key`, and `label_key` attributes (if it is not initially in this format, a custom collate function must be provided). The input data is processed sequentially batch by batch using a PyTorch `DataLoader`. At the end of its task, the `GoldVectorizer` returns a PyTorch `Dataset` via `vectorize_in_dataset` or a Pixeltable `Table` via `vectorize_in_table`. At the end, the vectors are stored within the column defined by the `vectorized_key` attribute. The output also includes the `idx` and `idx_vector` fields, with `idx` storing the index of the sample and `idx_vector` the index of the vector (a sample might be described by multiple vectors).

When the `Description` is not yet available and the processing pipeline requires the data to be vectorized, it is advised to instantiate the `TensorVectorizer` in the `GoldDescriptor` object of [Goldener](https://github.com/goldener-data/goldener). It avoids making new iterations with data loading and storage actions, and finally makes the whole operation faster.

```python
# Create a vectorizer to split the samples as vectors
# Here it keeps only the 1st vector in the description for all samples
keep = Filter2DWithCount(
  filter_count=1,
  filter_location=FilterLocation.START,
  keep=True,
)
vectorizer = TensorVectorizer(
  keep=keep,
  fusion_strategy=EmbeddingFusionStrategy.AVERAGE,
  transform_y=None,
  channel_pos=1
)

# Vectorize and store all the vectors
gold_vectorizer = GoldVectorizer(
  table_path="my_table_for_vectorization",
  vectorizer=vectorizer,
  data_key="embeddings",
  target_key="target",
  label_key="label",
  vectorized_key="vectorized",
)
vectorized_table = gold_vectorizer.vectorize_in_table(my_dataset)
```

### 2.4. Data selection with coreset

The main entry point in the code is [GoldSelector](https://github.com/goldener-data/goldener/blob/main/goldener/select.py).

Multiple algorithms exist to select some data based on the `Description` (vectors) of the samples. [Goldener](https://github.com/goldener-data/goldener) proposes different selection tools inheriting from the `GoldSelectionTool` class, for instance:
* `GoldGreedyKernelPointsSelectionTool` based on the `GreedyKernelPoints` class from [Coreax](https://coreax.readthedocs.io/en/latest/index.html)
* `GoldGreedyKCenterSelectionTool` based on an internal reimplementation of the Greedy K-Center algorithm
* `GoldZCoreSelectionTool` based on an internal reimplementation of the ZCore method
`GoldSelectionTool` assumes that all the data to select from fits in memory (RAM or VRAM). Those selection tools take the vectorized data as input and return a list of integers specifying which vectors have been selected.

In [Goldener](https://github.com/goldener-data/goldener), the `GoldSelector` class orchestrates the selection of samples from a `GoldSelectionTool`. In `GoldSelector`, the internal selection `Table` is created and partly populated (all columns except the selection one) before applying any selection algorithm. Then, the selection process is stratified between the classes when the `label_key` is provided. When all the vectors cannot be processed at once, the user can specify a chunk size to split the selection process in multiple smaller chunks of data. The assignment of the vectors to a given chunk is done randomly. Additionally, when the memory constraints are very restrictive, `GoldSelector` can integrate a `GoldReductionTool` in order to reduce the dimensions of the vectors. The available reduction tools are UMAP, PCA, t-SNE, and GaussianRandomProjection. An interface to use any PyTorch module as `GoldReductionTool` is also available.

Depending on the dataset and `Description`, multiple vectors can come from the same sample. Thus, in [Goldener](https://github.com/goldener-data/goldener), the selection algorithm might be applied multiple times before reaching the requested number of samples. This selection target can be specified as a fixed integer value or a float corresponding to a ratio of the full dataset. The indices of the selected vectors are used to populate the selection column with the specified value. Once a vector of a sample is selected, its other vectors are removed from the next selection round.

In [Goldener](https://github.com/goldener-data/goldener), `GoldSelector` is callable from either a PyTorch `Dataset` or a Pixeltable `Table`. If a `Dataset` is provided, each sample is expected to be accessible as a dictionary containing the keys defined in `vectorized_key`, `label_key`, `idx`, and `idx_vector` attributes (if it is not initially in this format, a custom collate function must be provided). During the initialization of the selection `Table`, the input data is processed sequentially batch by batch using a PyTorch `DataLoader`. Then, the data is selected from the internal Pixeltable `Table`, label by label if the `label_key` is provided, and chunk by chunk if required. Finally, the `GoldSelector` returns a PyTorch `Dataset` via `select_in_dataset` or a Pixeltable `Table` via `select_in_table`. At the end, the selection status is stored within the column defined by the `selection_key` attribute. The output also includes the `idx` and `idx_vector` fields, with `idx` storing the index of the sample and `idx_vector` the index of the vector (each sample might be described by multiple vectors).

```python
# Create a selection tool to select the samples
selection_tool = GoldGreedyKCenterSelectionTool(
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
selection_table = gold_selector.select_in_table(my_dataset, 0.5, "train")
```

## 3. Smart split

The main entry point in the code is [GoldSplitter](https://github.com/goldener-data/goldener/blob/main/goldener/split.py)

In [Goldener](https://github.com/goldener-data/goldener), a smart split is defined as a split based on:
* Vectors characterizing the samples/elements from both local and global semantics.
* A selection algorithm relying on vectors to gather together different samples based on their relative difference/distribution.
As opposed to a random split relying on randomness to split the data, a smart split makes the split more coherent with the target. This coherence is driven by the selection algorithm.

In [Goldener](https://github.com/goldener-data/goldener), the vectors characterizing the data can be obtained from a `GoldDescriptor` object and the selection of the samples per split from a `GoldSelector` class. The `GoldSplitter` class unifies these two objects in order to orchestrate the split on the data with one call. Its initialization requires the specification of the sets across which the data is split. This specification must ensure that the full dataset is split among the different sets. The population for each set can be a ratio or specified count. All sets must follow the same way to specify their target population.

In [Goldener](https://github.com/goldener-data/goldener), the `GoldSplitter` is callable from either a PyTorch `Dataset` or a Pixeltable `Table`. If a dataset is provided, each sample is expected to be accessible as a dictionary aligned with the descriptor requirements (if it is not initially in this format, a custom collate function must be provided). The input data is processed sequentially by the descriptor and then the selector is called sequentially for each of the specified sets (a sample already selected for a previous set is no longer accessible). At the end of its task, the GoldSplitter returns a PyTorch `Dataset` via `split_in_dataset` or a Pixeltable `Table` via `split_in_table`. The set affiliation status is stored within the selection column defined by the selector.

 ```python
gold_descriptor = GoldDescriptor(...) # descriptor with vectorizer included
gold_selector = GoldSelector(...)
gold_splitter = GoldSplitter(
    sets=[GoldSet("train", 0.7), GoldSet("val", 0.3)],
    descriptor=gold_descriptor,
    selector=gold_selector,
)

split_table = gold_splitter.split_in_table(dataset)
splits = gold_splitter.get_split_indices(
    split_table, selection_key="selected", idx_key="idx"
)
train_indices = splits["train"]
val_indices = splits["val"]
```

## 4. Bibliography

The readers of this post might be interested in the following resources:

* Moser, Brian B., et al. A coreset selection of coreset selection literature: Introduction and recent advances. arXiv preprint arXiv:2505.17799. 2025.
* Joseph, V. Roshan, et al. SPlit: An optimal method for data splitting. Technometrics 64.2: 166-176. 2022.
* Griffin, Brent A., et al. Zero-Shot Coreset Selection via Iterative Subspace Sampling. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2026.

## 5. Authors

[Yann Chéné, PhD,](https://huggingface.co/Yann-CV) is a Machine Learning (ML) engineer currently working at [Scortex](https://scortex.io/) - a company leveraging computer vision to automate manufacturing quality control. Within Scortex, he is involved in tasks from research to product integration and MLOps. His current focus is on improving the state of the art in image anomaly detection. Yann is also the creator of [Goldener](https://pypi.org/project/goldener/), an open source Python data orchestrator. Goldener proposes features to sample, split, organize, annotate, and curate data based on model embeddings/features in order to make the full ML lifecycle more efficient.


## 6. Miscellaneous

Sponsored by [Pixeltable](https://www.pixeltable.com/): Multimodal Data, Made Simple. Video, audio, images, and documents as first-class data types, with storage, orchestration, and retrieval unified under one table interface.
