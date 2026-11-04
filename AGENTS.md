# Repository guidance

## Project map

Goldener is a Python library for data-centric AI pipelines. Embeddings describe
samples; vectorization produces vectors; reduction, clustering, selection, splitting,
and batch organization operate on those representations. PixelTable persists
intermediate results so operations can resume without recomputing completed work.

- `goldener/embed.py`: extract semantic representations of data.
- `goldener/vectorize.py`: turn representations into vectors for downstream actions.
- `goldener/reduce.py`: reduce representation dimensionality.
- `goldener/describe.py`: orchestrate dataset description.
- `goldener/clusterize.py`: group similar data.
- `goldener/select.py`: choose representative subsets.
- `goldener/split.py`: partition data into sets.
- `goldener/do.py`: define the common foundation for orchestration objects.
- `goldener/organize.py`: balance training batches using data clusters.
- `goldener/torch_utils.py`: support PyTorch data handling.
- `goldener/pxt_utils.py`: support persistence and retrieval of results.
- `goldener/utils.py`, `goldener/distance.py`: support shared data operations.
- `goldener/vision/`: adapt processing to vision data.
- `tests/`: verify behavior; also read `tests/AGENTS.md`.
- `README.md`: explain usage and development.
- `scripts/generate_api_docs.sh`: generate API documentation.

## Architecture

- `GoldDoer` objects orchestrate an action over data, using `Gold...Tool` objects
  to perform the actual computation. Orchestration owns data access, batching,
  persistence, restart behavior, and composition of pipeline stages.
- `Gold...Tool` objects perform the actual operations on PyTorch tensors. Keep
  algorithm behavior here, independently of dataset traversal and persisted tables.
  External backend adapters expose the same tensor-facing contracts.
- When extending a feature, place computation in the appropriate tool and let the
  doer coordinate its use. Reuse tools across orchestration paths.

## Code style

- Support Python >=3.11; CI currently runs Python 3.13. Use four-space indentation,
  Ruff formatting, `snake_case` functions/variables and `PascalCase` classes.
  Follow existing `Gold...Tool` and `Gold...` orchestration naming conventions.
- Annotate new and changed library functions, including return types and `-> None`
  for constructors. Prefer modern `list[T]`, `dict[K, V]`, and `T | None` syntax.
  Avoid unrelated annotation rewrites.
- Use `ABC`/`abstractmethod` for algorithm interfaces, dataclasses for data/config
  containers, enums for strategies, and properties where the existing API validates
  mutable configuration. Preserve public argument names, defaults, and return types.
- Reuse existing tensor, sampling, collation, and PixelTable helpers before adding
  another implementation. Keep tensor algorithms separate from persistence and
  DataLoader orchestration. Keep modality-specific behavior in `goldener/vision/`.
- Validate invalid public inputs with informative exceptions, normally `ValueError`.
  Use `NotImplementedError` for unsupported capabilities, following the relevant API.
  Use module-level loggers for operational messages and warnings.
- Keep changes focused. Do not silently change device, dtype, shape, index meaning,
  persistence behavior, or randomness while adding an algorithm or option.
- When exposing a new root-level public API, update both imports and `__all__` in
  `goldener/__init__.py`. Follow the existing distinction between module-only and
  root-level exports; not every class is exported at the root.

## Docstrings

- Use Google-style triple-double-quoted docstrings, as required by the pdoc build.
  Start with a short description of behavior, then explain non-obvious semantics.
- Public modules describe their role and main APIs; public classes describe their
  purpose and relevant configuration/state. Document public methods and helpers.
  Document private helpers when their behavior or invariants need explanation.
- Use `Args:`, `Returns:`, `Raises:`, and `Attributes:` where relevant, with indented
  entries. A trivial property or helper can have a one-line docstring. Prefer a
  separate indented `Returns:` description when adding or expanding documentation.
- Explain tensor shapes and axis meanings, index domains, optional values, defaults,
  count-versus-ratio semantics, and side effects when relevant. Clarify whether
  results are sample indices, vector indices, cluster assignments, or embeddings.
- Docstrings must agree with implementations, signatures, tests, and related APIs.
  Verify shapes, defaults, method names, side effects, and supported capabilities.
  Fix inconsistencies when encountered; do not preserve or copy incorrect prose.
- Update docstrings with API changes and README examples when usage changes. Do not
  commit generated `docs/api/` output.
- Avoid unnecessary comments in both library code and tests. Add a comment only
  when essential context, intent, or a non-obvious constraint cannot be conveyed
  clearly by names and code structure. Do not narrate statements or add routine
  arrange/act/assert labels. Keep required API docstrings distinct from inline comments.

## Rules by API family

### Embedding tools (`goldener/embed.py`)

- Implement `GoldEmbeddingTool.embed(...) -> dict[str, torch.Tensor]` and
  `embed_and_fuse(...) -> torch.Tensor`. Preserve stable layer keys; multimodal keys
  follow `"{modality}.{layer}"`.

### Tensor vectorization and vision factories

- `GoldTensorVectorizationTool.vectorize(x, y=None)` returns `Vectorized`, with 2D
  `vectors` and one `batch_indices` entry per vector identifying its source sample
  within the input batch. Keep these aligned through filtering and fusion.
- This API requires inputs of at least three dimensions and respects `channel_pos`;
  do not confuse it with downstream tools that require already-vectorized 2D inputs.
- Preserve processing order: keep/remove filters, target filtering, random filtering,
  optional selection, then fusion. All-zero targets currently skip target filtering.
- Preserve integer-count versus float-ratio behavior for `in_selection_size` and the
  rejection of CONCAT fusion, which changes feature width.
- Vision factories should configure the shared vectorizer/filter API. ViT tokens use
  `channel_pos=2`; preserve prefix/class/patch ordering and generator forwarding.

### Reduction tools (`goldener/reduce.py`)

- Subclass `GoldReductionTool` for `transform(x)`; subclass
  `GoldReductionToolWithFit` when also implementing `fit(x)` and `fit_transform(x)`.
- Call the inherited `_validate_input` for 2D `(num_vectors, feature_dim)` inputs.
  Return transformed tensors with the same row correspondence.
- Wheb required, detach and move inputs to CPU for NumPy conversion;
  restore the input device and dtype on transformed outputs. Preserve the warning
  and `fit_transform` fallback when an estimator such as TSNE lacks `transform`.

### Clustering tools (`goldener/clusterize.py`)

- Implement `GoldClusteringTool.fit(x, n_clusters)` and `predict(x)`, validating 2D
  inputs with `_validate_input`. `fit` returns assignments, not the estimator itself;
  assignments are a 1D tensor with one cluster index per input row.
- when required the tool converts inputs via detached CPU NumPy arrays, and returns labels with `torch.from_numpy`.
  Do not assume it has the reduction adapter's device/dtype restoration contract.

### Selection tools (`goldener/select.py`)

- Implement `GoldSelectionTool.select(x, k, anchors=None) -> list[int]`. Results are
  row positions in `x`, not selected vectors or original dataset IDs.
- Use `_validate_input` for 2D inputs and preserve the optional 2D anchors contract
  with matching feature width. Preserve each algorithm's bounds and anchor behavior.
- Reuse `DistanceType`/`distance_func` for configurable distances where applicable.
  Preserve each implementation's seed/generator controls rather than adding global
  random-state changes.

### Persisted orchestration

- `GoldDescriptor`, `GoldVectorizer`, `GoldClusterizer`, and `GoldSelector` derive
  from `GoldDoerWithTable`; `GoldSplitter` derives from `GoldDoer` and composes stages.
  Put shared initialization in the appropriate base and forward options via `super()`.
- Preserve paired entry points: `describe_in_*`, `vectorize_in_*`, `cluster_in_*`,
  `select_in_*`, and `split_in_*`, with `*_in_table` returning PixelTable tables and
  `*_in_dataset` returning the existing dataset wrapper. Follow each method's actual
  supported Dataset/Table input types and keyword defaults.
- Respect configurable column keys, `to_keep_schema`, shape metadata, and primary
  keys. Keep `idx` (source sample) distinct from `idx_vector` (individual vector).
  Multiple vectors/labels may belong to one sample; preserve that correspondence.
- Preserve restart/idempotence, `allow_existing`, `max_batches`, and documented
  `drop_table` behavior. Reuse `pxt_utils` table/schema/query/conversion helpers.
  Do not eagerly drop tables still needed by a returned dataset.
- Preserve `collate_keeping_sequences_as_sequences` as the shared default and honor
  custom `collate_fn`. Sequence-valued labels must remain associated with samples.
- Distributed processing is currently unsupported in the table doers. Keep explicit
  rejection when `distribute=True`, including property setters; do not silently accept it.
- For split/selection changes, retain count-versus-ratio validation, label handling,
  exclusions, and existing-selection semantics. Reuse `GoldSet` and shared sampling
  helpers. Check composed stages when forwarding settings.

### Dataset adapters and batch sampling

- Keep `GoldPxtTorchDataset` consistent with stored array shape metadata and the
  existing table-to-dataset conversion and cleanup behavior. Preserve PyTorch
  iteration/reset semantics in `ResetableTorchIterableDataset`.
- `GoldClusterizedBatchSampler.__iter__` yields lists of dataset indices. Preserve
  Subset index remapping, cluster exhaustion strategies, shuffle/generator behavior,
  and the documented balancing/oversampling semantics.

## Validation

- Use the existing environment; setup is `uv sync --all-extras` when needed.
- Run affected tests first: `uv run pytest tests/test_<module>.py` (or a focused
  node). Shared API changes require exercising affected implementations and callers.
- Relevant CI checks are `uv run pytest tests/`, `uv run mypy .`,
  `uv run ruff check .`, and `uv run ruff format --check .`.
- For docstring/rendering changes, use `./scripts/generate_api_docs.sh` when needed.
  A change limited to Markdown agent instructions does not require the Python suite.
- Report what was checked and any blockers; never claim unrun checks passed. Before
  finishing, review the diff for unintended changes and test redundancy as required
  by `tests/AGENTS.md`.
