<picture class="github-only">
    <img
        alt="Goldener Logo"
        src="https://raw.githubusercontent.com/goldener-data/goldener/main/docs/statics/goldener_brand.png"
        width="70%"
    />
</picture>

A python library orchestrating data during the full life cycle of machine learning pipelines.

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Package](https://img.shields.io/pypi/v/goldener?color=6D165C)](https://pypi.org/project/goldener/)

[**Overview**](#overview) |
[**Principles**](#key-design-principles) |
[**Features**](#example-of-features) |
[**Installation**](#installation) |
[**Contribute**](#contribute)

# Overview

Goldener is an **open-source Python library** (Apache 2 licence) designed to manage the **orchestration
of data** (sampling, splitting) during the full life cycle of machine learning (ML) pipelines.

In the artificial intelligence (AI) era, the data is the new gold. Being able to collect it is already something
but **creating value from it is the real challenge**. Goldener is designed to help to make the most of the available data.
It provides tools to orchestrate data during the full life cycle of machine learning pipelines,
from the training phase to the monitoring phase.

Goldener makes the **right data** available at the **right time**, allowing to **optimize the performance**
of any ML pipelines while **minimizing the costs** (time, performance, computing resources) of data sampling and labeling.

When it's time to annotate data, Goldener find the most representative subset to annotate. During annotation, it can help
to define annotation guidelines by spotting specific cases or as well run annotation quality checks.
Once enough data is annotated, Goldener can split it in multiple sets (train, validation, test) ensuring the reproduction
of the task variability. During the training phase, Goldener can balance efficiently the data
to optimize the training time and the model performance. Finally, when the model is deployed, Goldener can find
the most informative data to monitor the model performance and detect any drift in the data distribution.

# Key design principles

Goldener is designed to process large datasets efficiently. It is built on the assumption
that every AI lifecycle is most of the time iterative and incremental. Its design principles are:

- **Progressive batch processing**: Each task can be stopped and restarted on demand (or failure).
Already computed results are not recomputed.
- **Multipurposes embeddings**: The same embeddings are used for
the different for different tasks (selection, splitting, monitoring, etc.).
- **Modality-agnostic**: The same tool is actionable for any data modalities (text, image, video, tabular, etc.)
and even for multimodality data.

This is not yet applied but for the next iterations, the following principles will be as well followed:

- **Distributed first**: Any task can be distributed across multiple machines.
- **On demand access to pipelines**: All processing pipelines are serializable.
They are stored and available whenever a new request is made.


# Example of features

## Sampling among not annotated data

Goldener can find the most representative data subset to annotate. It can extract and store semantic knowledge of the data
from embeddings extracted with pre-trained models. Then, it leverages this knowledge to find the most representative
subset of data to annotate. This subset of data can be annotated in order to train or monitor a model.

```python
from goldener import (
    GoldSelector,
    GoldDescriptor,
    TorchGoldFeatureExtractor,
    TorchGoldFeatureExtractorConfig,
    TensorVectorizer,
)

gd = GoldDescriptor(
    table_path="my_table_for_description",
    extractor=TorchGoldFeatureExtractor(
        TorchGoldFeatureExtractorConfig(
            model=my_model,
            layers=my_layers,
        )
    ),
    vectorizer=TensorVectorizer()
)

gs = GoldSelector(
    table_path="my_table_for_selection", selection_key="selection"
)

description = gd.describe_in_table(dataset)
selection_table = gs.select_in_table(description, 100, "to_annotate")
selected = GoldSelector.get_selection_indices(selection_table, "to_annotate", "selection")

```

## Splitting annotated data in train and validation sets

Goldener can split data between the train and validation sets ensuring that the training set is containing
most of the different situations for the tasks. From a description of the samples (embeddings), the most different/unique
elements are kept for the training set while the least informative ones are kept for the validation set.

```python
from goldener import (
    GoldSet,
    GoldSplitter,
    GoldDescriptor,
    GoldSelector,
)

gd = GoldDescriptor(...) # reuse the descriptor used for smart sampling
gselector = GoldSelector(...)
gs = GoldSplitter(
    sets=[GoldSet("train", 0.7), GoldSet("val", 0.3)],
    descriptor=gd,
    selector=gselector,
)

split_table = gs.split_in_table(dataset)
splits = gs.get_split_indices(
    split_table, selection_key="selected", idx_key="idx"
)
train_indices = splits["train"]
val_indices = splits["val"]

```

## Clustering data to define annotation guidelines

Among the data, there are often multiple "modes" (e.g. different types of images, different types of text, etc.).
Goldener can clusterize the data to find these different modes. Then, the different clusters can be leveraged
to define annotation guidelines for each cluster.

```python
from goldener import (
    GoldClusterizer,
    GoldSKLearnClusteringTool,
    GoldDescriptor,
    TorchGoldFeatureExtractor,
    TorchGoldFeatureExtractorConfig,
    TensorVectorizer,
)
from sklearn.cluster import KMeans

gd = GoldDescriptor(...) # reuse the descriptor used for smart sampling
gcluster = GoldClusterizer(
    table_path="my_table_for_clusterization",
    clustering_tool=GoldSKLearnClusteringTool(KMeans(n_clusters=10)),
    cluster_key="cluster",
)

description = gd.describe_in_table(dataset)
clustered_table = gcluster.clusterize_in_table(description)

for cluster_id in range(10):
    cluster_indices = get_cluster_indices(clustered_table, "cluster", cluster_id)

    # sample few samples and use them to define annotation guidelines for this cluster

```

# Installation

Installing Goldener is as simple as running the following command:

```bash
pip install goldener
```

# Contribute

We welcome contributions to Goldener! Here's how you can help:

## Getting Started

1. Fork the repository
2. Clone your fork
3. Install the dependencies
4. Create your branch and make your proposals
5. Push to your fork and create a pull request
6. The PR will be automatically tested by GitHub Actions
7. A maintainer will review your PR and may request changes
8. Once approved, your PR will be merged

## Development

To set up the development environment:

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment (optional but recommended):
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
```

3. Install development dependencies:
```bash
uv sync --all-extras  # Install all dependencies including development dependencies
```

4. Run tests:
```bash
uv run pytest .
```

5. Run type checking with mypy:
```bash
uv run mypy .
```

6. Run linting with ruff:
```bash
# Run all checks
uv run ruff check .

# Format code
uv run ruff format .
```

7. Set up pre-commit hooks:
```bash
# Install git hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

The pre-commit hooks will automatically run:
- mypy for type checking
- ruff for linting and formatting
- pytest for tests

whenever you make a commit.

## Release Process

To release a new version of the `goldener` package:
1. Create a new branch for the release: `git checkout -b release-vX.Y.Z`
2. Update the version `vX.Y.Z` in `pyproject.toml`
3. Run `uv sync` to update the lock file with the new version
4. Commit the changes with a message like `release vX.Y.Z`
5. Merge the branch into `main`
6. Trigger a new release on GitHub with the tag `vX.Y.Z`
