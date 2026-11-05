# Test guidance

These instructions apply to all tests and supplement the root `AGENTS.md`.

## Mandatory reuse and duplication review

Before adding or editing tests:

1. Search the existing suite for the API, behavior, error message, and related API
   implementations (for example, `rg -n 'select|anchors|rejects' tests/`). Read the
   relevant tests and their setup, not just their names.
2. Identify existing coverage and the specific missing behavior. Check for exact
   redundancy and near duplication: tests with the same setup/action/assertions
   that differ only in implementation, shape, option value, or invalid input.
3. Prefer extending an existing test, parameter list, fixture, or small helper.
   Add meaningful assertions to an existing scenario when they verify that same
   behavior. Reuse existing dummy datasets/models and factories before creating more.
4. When cases share a contract, use `pytest.mark.parametrize` or a parametrized
   fixture with descriptive IDs instead of copying whole tests. Keep capability
   differences explicit rather than hiding them behind broad skips or conditionals.
5. Add a separate test only when strictly necessary to cover a missing behavior or
   regression that cannot be covered clearly by extending existing coverage.
   Reuse setup where practical and state
   the new coverage in the change summary. Do not add a new test file merely because
   a new class or option was added.

After changes, review the entire test diff against neighboring and related tests.
Consolidate redundancy introduced by the change while retaining meaningful checks.
In the final summary, briefly identify reused/extended coverage and why any separate
tests were needed. This review is required even when all tests pass; passing does
not establish that a test adds useful coverage.

## Existing places to extend

- Module tests generally live at `tests/test_<module>.py`; vision tests are in
  `tests/vision/`. Use the existing `Test<Class>` grouping or module functions.
- `tests/test_do.py` has a parametrized `make_doer` fixture for shared constructor
  defaults. Extend that coverage for applicable implementations or shared settings
  instead of duplicating default checks across orchestration test classes.
- `tests/test_embed.py` covers fusion and embedding tools; `tests/test_reduce.py`
  covers reduction adapters; `tests/test_clusterize.py` and `tests/test_select.py`
  cover both algorithm tools and persisted orchestration.
- `tests/test_vectorize.py` has vectorization tensor builders and filtering/selection
  cases. `tests/vision/test_vision_vectorizers.py` exercises factories through the
  shared vectorizer. Reuse these for new options and shapes.
- Description, split, and organization tests already have dummy models, datasets,
  and descriptor/vectorizer fixtures. Reuse local fixtures first. If multiple files
  need the same setup, extract a small focused helper or scoped `conftest.py` fixture;
  do not import tests from each other or build a general testing framework.

## Test quality and isolation

- Avoid unnecessary test comments and docstrings. Add them only when strictly
  necessary to explain context that the test name, setup, and assertions cannot
  convey, such as a non-obvious regression condition. Do not narrate test steps
  or add routine arrange/act/assert labels.
- Test observable behavior and contract invariants: tensor values/shapes, source
  indices, assignments, errors, or persisted results. Avoid tests that only mirror
  implementation details or repeat a dependency's own tests.
- A new algorithm sharing an interface should run applicable contract checks using
  the existing cases, plus only the algorithm-specific assertions it needs. Include
  invalid shape/boundary cases where relevant without copying them for every class.
- Use small CPU tensors and lightweight models; seed randomness or pass a local
  generator when asserting reproducibility. Use tolerant comparisons for floating
  point values and avoid timing-sensitive assertions.
- Use `pytest.raises(..., match=...)` for specific expected failures and `caplog`
  for meaningful warnings. Avoid broad exception catches and new arbitrary sleeps.
- Pure tensor tests should not need PixelTable. Integration tests should exercise
  real persistence when persistence is the behavior being checked.
- Most orchestration PixelTable tests use `unit_test.*` tables with setup/teardown
  that recreate and remove the `unit_test` directory. Follow the cleanup pattern
  already used by the relevant module, use distinct table names, and never point
  test cleanup at user tables.
- Do not parallelize tests sharing a directory without first providing isolated
  per-worker namespaces.
- For persistence changes, extend existing restart, `allow_existing`, `max_batches`,
  schema, index-mapping, or `drop_table` scenarios as appropriate. Keep regression
  assertions that distinguish correct behavior from the original failure.
- Run the affected tests, including all cases of any changed parametrization and
  callers of shared fixtures. Broaden to related modules for shared API changes.
  Do not expand unrelated test coverage just to increase test count.
