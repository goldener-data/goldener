![goldener](docs/statics/goldener_brand.png)

# Goldener - Make your data even more valuable

Goldener is an open-source Python library (Apach 2 licence) designed to manage the orchestration 
of data (sampling, splitting and labeling) during the full life cycle of machine learning pipelines. 

In the artificial intelligence (AI) era, the data is the new gold. Being able to collect it is already something.
However, using blindly all these data is for sure costly: annotation cost, storage cost, training cost. Goldener is 
aiming to reduce all these costs by optimizing:
- Data sampling and splitting: sample the right data to best train and test your machine learning models.
- Data labeling: Ensure high-quality data to train and test machine learning models.

## Sampling and labeling in AI life cycle

Successful machine learning pipelines are all about data. All along the Machine Learning (ML) life cycle, getting access
to data fully modelizing the real world task is key. Both training and test data are then crucial to the success of
the ML pipelines and are continuously updated to ensure the pipeline performances during its whole usage:

- **Training**: The training data defines the model ability to succeed its task.
    Training from not representative enough data makes the model unable to learn the task adequately. In the meantime,
    using too much data will slow down the training process (time and money lost). Once the model is deployed, 
    selecting new training data efficiently is as well crucial to solve data drift issues.
- **Test**: The test data drives the design and validation of tested pipelines. Testing from not representative enough data 
    ends up with bad design decision leading to poor performances in production. Once the model is deployed, selecting
    new testing data is as well crucial to monitor the model performances and ensure it succeeds its task.

In the meantime, all the sampled data is required to be labelled in order to be used in the ML pipelines, at least for
the test/monitoring sets in case of unsupervised learning. Labeling data is costly and time-consuming, 
especially when it comes to large datasets. Most of the time, it involves an iterative process including human
labelers potentially helped by some AI tools. An efficient data sampling before labeling allows to optimize
the time and cost to access new labeled data. In the meantime, depending on the task and target, the bad quality of the 
labeled data can:
- Lead to a model unable to learn the task adequately (wrong labels pushing it in the wrong direction)
- Lead to wrong design decisions during the model validation phase (wrong test data leading to wrong conclusions)

The goal of Goldener is to provide the orchestration ensuring the access to high-quality and representative enough data 
during the whole life cycle of the ML pipelines. User gets the right data at the right time, ensuring the best performances
of the ML pipelines while minimizing the costs of data sampling and labeling.

<div align="center">
    <img src="docs/statics/goldener_ai_lifecycle.png" alt="Sampling and labeling during AI lifecycle" width="600"/>
</div>


## Goldener for sampling and labeling orchestration

As a gardener exploiting the most of a good ground, Goldener aims to make the most of your gold (data) 
and make it even more valuable. Mainly, Goldener features a set of tools to help you to:

- **Gold prospection**: Sample and split the most valuable gold (data)
    - Sample among raw data: Spot the data allowing to modelize the task (pipeline training) 
    or spot weaknesses of a deployed pipeline while minimizing the need for labeling.
    - Split labeled data: Ensure enough representativeness in both the training and test sets 
    while optimizing the training process in effectiveness and time.

- **Gold refining**: Ensure the gold (data) quality
    - Assist in the labeling processing: Make human labeling faster with some smart labeling tools 
    (for instance create image segmentation masks from a single click).
    - Label data automatically: Propose labels for raw data based on foundation models or existing labeled data.
    - Curate newly labeled data: Identify potential labeling mistakes allowing humans labelers to converge 
    toward high quality labeled data.


## Key design principles

Goldener is designed in order to be able to process large datasets efficiently and effectively. It is integrating
that every AI lifecycle is most of the time iterative and incremental. The design principles are:

- **Progressive batch processing**: Each task can be stopped and restarted on demand (or failure). 
Already computed results are not recomputed.
- **Distributed first**: Any task can be distributed across multiple machines. 
- **On demand access to pipelines**: All processing pipelines are serializable. 
They are stored and available whenever a new request is made.
- **Multipurposes embeddings**: Whenever it is possible, the same embeddings are used for
the different prospection and refining actions on the same data.

To orchestrate both the sampling and labeling of data in Goldener, the same data is moving from steps to steps
during the AI lifecycle. In addition, the information gathered all along the cycle is valuable to drive the efficiency
of the next sampling and labeling. Thus, all the data is cached behind the scene and accessible any time.

<div align="center">
    <img src="docs/statics/goldener_data_workflow.png" alt="Data workflow in Goldener" width="600"/>
</div>

## Current focus

Goldener is a work in progress and is currently in the early stages of development.
The main focus is on building a robust and efficient framework for image data. However, every design choice
is made to be as much as possible multimodal. The goal is to make Goldener a versatile tool
that can be adapted to various data types in the future.

## Main features

We are still working on the 1st one !

## Installation

1. Clone the repository:
```bash
git clone https://github.com/goldener-data/goldener.git
cd goldener
```

2. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and activate a virtual environment (optional but recommended):
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
```

4. Install the package and its dependencies:
```bash
uv sync
```

## Configuration

### Logging

The logging level can be configured through environment variables:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export GOLDENER_LOG_LEVEL=INFO
```

Default log level: WARNING

## Contributing

We welcome contributions to Goldener! Here's how you can help:

### Getting Started

1. Fork the repository
2. Clone your fork
3. Install the dependencies
4. Create your branch and make your proposals
5. Push to your fork and create a pull request
6. The PR will be automatically tested by GitHub Actions
7. A maintainer will review your PR and may request changes
8. Once approved, your PR will be merged

### Development

To set up the development environment:

1. Install development dependencies:
```bash
uv sync --all-extras  # Install all dependencies including development dependencies
```

2. Run tests:
```bash
pytest .
```

3. Run type checking with mypy:
```bash
uv run mypy .
```

4. Run linting with ruff:
```bash
# Run all checks
uv run ruff check .

# Format code
uv run ruff format .
```

5. Set up pre-commit hooks:
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
