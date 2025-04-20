# Goldener - Make your data even more valuable

Successful machine learning pipelines are all about data. All along the Machine Learning (ML) life cycle, get access
to data closed enough to the real world distribution is key. Both training and test data are crucial to the success of
the ML pipeline and are continuously updated to ensure the model performances during its whole usage.

- **Training**: The training data define the model ability to succeed its task.
    Not representative enough data make the model unable to learn this task. In the meantime,
    using too much data will slow down the training process (time and money lost). Adapting to data drift
    will be possible by making new training cycles from new data.
- **Test**: The test data is driving the pipeline design and validate its potential
    before its release. Not representative enough data end up with bad design decision leading to poor performances
    in production. Ensuring a continuous correct behavior will be possible by making new test analysis from new data.

In the artificial intelligence (AI) era, the data is the new gold. Being able to collect it is already something.
Using blindly all this data is for sure costly (annotation cost, storage cost, training cost), and even though it is counterintuitive it can lead to
bad performances (wrong train/test balance).

As a gardener exploiting the most of a good ground, Goldener aims to make the most of your gold (data) and make it even more valuable.
Mainly, Goldener features a set of tools to help you to:

- **Prospect**: Find the most valuable gold nuggets
    - Sample the right data to train your model, representative enough to ensure performances
        but as well small enough to be efficient
    - Sample the right data to test your model, representative enough to cover all possible data
        but as well with right distribution to ensure good performances in real life.

- **Refine**: Ensure your gold quality
  - Identify issues among existing annotations.
  - Ensure high quality annotation processes.
  - Propose new annotations to sampled data.

In our competitive world, taking too much time to leverage your data is not an option anymore. Goldener is designed to
be fast and efficient. All the proposed tools are based on 2 main building blocks:

- **GoldStorage**: Keep your gold safe and ready to be valuable
  - Store the data and all its features to ensure efficient and fast processing.
  - Persist pipelines allowing to prospect and refine the data.

- **GoldOrder**: Easily order prospection and refining processes
  - All the tools are designed to handle the same type of inputs.
  - All the tools are designed to be used in a distributed way.


## Current focus

Goldener is a work in progress and is currently in the early stages of development.
The main focus is on building a robust and efficient framework for image data. However, every design choice
is made to be as much as possible multimodal. The goal is to make Goldener a versatile tool
that can be adapted to various data types in the future.

## Main features

- In progress: [Illustrate the class from samples](https://github.com/goldener-data/goldener/wiki/Help-humans-refining-your-gold-(data)#illustrate-the-class-from-samples)

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
uv sync --all-extras  # Install all dependencies including development dependencies
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
uv pip install -e ".[dev]"
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
