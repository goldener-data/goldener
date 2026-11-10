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
- `Gold...Tool` objects that use randomness must expose public methods to get and
  set their complete random state, including generators supplied by callers and
  randomness in composed tools or backends. Getting state must return an independent
  snapshot without advancing randomness; restoring it must reproduce subsequent
  random choices. A seed alone is not a snapshot of an advanced generator's state.
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
- Reuse existing utility functions for runtime type checks when available. Extract
  checks used in multiple places into a utility function, and consider whether a
  new check is likely to be needed elsewhere when deciding where it belongs.
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
- Document public attributes in an `Attributes:` section of the class docstring,
  explaining their meaning and any relevant defaults or constraints.
- Include `Args:`, `Returns:`, and `Raises:` sections in public function and
  method docstrings when they contain relevant information, with indented entries.
  Document no arguments, no return value, or no expected exceptions in the prose only
  when that clarification is useful.
  methods are exempt from these required sections; a trivial private helper can
  have a one-line docstring.
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

### Module docstrings

- Start each public library module with a top-level docstring before imports.
  Open with a short statement of the module's main goal, then explain its role
  in the data pipeline and when its functionality is used.
- Follow the introduction with a concise bulleted overview of the `GoldDoer`
  subclasses and `Gold...Tool` classes implemented in the module, including its
  abstract interfaces and shared base classes. Name each class in backticks and
  explain its responsibility; group closely related implementations when useful.
  Distinguish tensor computation tools from dataset/table orchestration and explain
  how the concrete implementations relate to their base classes.
- Include supporting public configuration classes, enums, or helpers when needed
  to understand the main APIs. Link related stages using qualified names such as
  `goldener.describe.GoldDescriptor`, distinguishing those references from classes
  implemented in the module. Modules without doers or tools should introduce their
  own main APIs rather than inventing a doer/tool overview.
- Use the module docstrings in `goldener/select.py`, `goldener/clusterize.py`, and
  `goldener/embed.py` as references for structure and level of detail. Keep the
  overview consistent when adding, removing, or renaming classes; leave detailed
  parameters and algorithm contracts to class and method docstrings.

### GoldDoer and Gold...Tool class docstrings

- Follow the content and order of existing class docstrings in the same API family.
  Use `GoldDescriptor`, `GoldVectorizer`, `GoldClusterizer`, and `GoldSelector` as
  references for persisted doers; use `GoldSplitter` for composed orchestration.
  For tools, use `GoldTorchEmbeddingTool`, `GoldTensorVectorizationTool`, and the
  concrete selection tools as references. Match the structure, but verify every
  claim against the class being documented rather than copying implementation details.
- Start with a one-line description of the action and its data or result. Follow
  with explanatory paragraphs, then an indented Google-style `Attributes:` section
  for public configuration and state, including relevant inherited attributes.
  Distinguish constructor inputs documented in `Args:` from stored attributes
  and properties documented in `Attributes:`.
- For `GoldDoer` subclasses, explain the supported Dataset/Table inputs, the tools
  or pipeline stages used, and the outputs produced. For persisted doers, describe
  local PixelTable storage, reuse of completed results on repeated or resumed calls,
  batching or chunking, tensor-to-array conversion when applicable, and the `idx`
  and `idx_vector` columns actually used to associate results with inputs. State
  distribution support accurately. For composed doers, explain stage order,
  optional stages, and how samples are assigned or passed between stages instead
  of implying that the doer owns a result table.
  - For persisted doers, explain the computation tools, table path and column keys,
    collation and retained schema, batch/chunk settings, and applicable lifecycle
    controls such as `allow_existing`, `drop_table`, and `max_batches` when present.
    For composed doers, document their own configuration and clarify which settings
    are forwarded to stages. Include defaults, constraints, and scope where needed,
    especially whether an option applies only to a dataset entry point or is forwarded.
  to a dataset entry point or is forwarded to composed stages.
- For `Gold...Tool` classes, explain the tensor operation or algorithm, relevant
  input/output shapes and index meanings, and non-obvious behavior such as fitting,
  fusion/filter order, anchor support, backend limitations, or model hooks. Describe
  mutable versus fixed configuration and random-state behavior when relevant.
  Use `Attributes:` for the tool's configuration and state; keep dataset traversal
  and table lifecycle details in the doer documentation. Abstract interfaces and
  simple adapters can stay brief when their contracts are already documented by
  the base class and public methods.

## README feature documentation

- When adding a new feature, add or update its section in `README.md` as part of
  completing the feature.
- Start with the problem the feature solves and a brief description of what it
  does and when to use it. Keep the explanation user-focused and omit internal
  implementation details; leave technical reference details to API documentation.
- Include a short usage example illustrating the main use case through the public
  API. Show the essential setup and call, and briefly explain the result.
- Use existing feature sections in `README.md` as inspiration for structure, tone,
  and example length. Ensure the example matches the implemented API.

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
- When required, detach and move inputs to CPU for NumPy conversion;
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
- Always implement `*_in_dataset` by calling `*_in_table` and converting its result
  to a dataset. Keep one computation path for both entry points.
- Save intermediate results in local tables wherever possible. Computation must be
  restartable: stopping and resuming an operation must reuse completed elements
  instead of recomputing them. Repeated calls must not duplicate completed work.
- Provide `estimate_computation_time` when feasible. Factor shared computation out
  of the main operation and reuse it for estimation. Estimation must leave random
  state unchanged, including any supplied generators, so it does not affect the
  subsequent computation.
- Expose a distribution capability even when its implementation is pending.
  When unsupported, default `distribute` to False and reject `distribute=True`
  explicitly, both at construction and through mutable configuration.
- Respect configurable column keys, `to_keep_schema`, shape metadata, and primary
  keys. Every sample must be linked to an `idx`. Every vector must retain its source
  sample's `idx` and have an `idx_vector` identifying the individual vector uniquely.
  Multiple vectors/labels may belong to one sample; preserve that correspondence.
- Provide a result retrieval function keyed by `idx` or `idx_vector`, as appropriate
  to the operation, so callers can associate results with samples or individual vectors.
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

- Do not add tests for GitHub Actions workflows unless they address a concrete risk
  to library usage, such as broken installation, release artifacts, or runtime
  behavior. When this exception applies, test the affected user-facing behavior.
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
