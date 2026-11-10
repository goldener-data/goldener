import pytest
import torch
import numpy as np
import pixeltable as pxt
from pixeltable import Error
from torch.utils.data import Dataset

from goldener.novelty import (
    GoldNearestNeighborNoveltyScoringTool,
    GoldNoveltyScorer,
)
from goldener.reduce import GoldReductionToolWithFit
from goldener.select import DistanceType


@pytest.fixture
def clean_unit_test():
    pxt.drop_dir("unit_test", force=True)
    pxt.create_dir("unit_test", if_exists="ignore")
    yield
    pxt.drop_dir("unit_test", force=True)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (DistanceType.COSINE, [0.0, 1.0, 1.0]),
        (DistanceType.EUCLIDEAN, [3.0, 1.0, 0.0]),
    ],
)
def test_nearest_reference_scores_and_blocks(distance, expected):
    reference = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    queries = torch.tensor([[4.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    tool = GoldNearestNeighborNoveltyScoringTool(distance, distance_batch_size=1)
    tool.fit(reference)

    scores = tool.score(queries)

    assert scores.tolist() == pytest.approx(expected)
    assert scores.dtype == queries.dtype
    assert scores.device == queries.device
    unblocked = GoldNearestNeighborNoveltyScoringTool(distance, distance_batch_size=8)
    unblocked.fit(reference)
    assert torch.equal(unblocked.score(queries), scores)
    assert torch.equal(
        torch.cat([unblocked.score(part) for part in queries.split(1)]), scores
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_score_restores_query_dtype(dtype):
    tool = GoldNearestNeighborNoveltyScoringTool()
    tool.fit(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    query = torch.tensor([[0.0, 1.0]], dtype=dtype)

    result = tool.score(query)

    assert result.dtype == dtype
    assert result.device == query.device
    assert result.item() == pytest.approx(1.0)


def test_fit_clones_reference_and_accepts_empty_queries():
    reference = torch.tensor([[1.0, 2.0]])
    reference.requires_grad_()
    tool = GoldNearestNeighborNoveltyScoringTool()
    tool.fit(reference)
    with torch.no_grad():
        reference.zero_()

    assert tool.reference_bank.tolist() == [[1.0, 2.0]]
    assert not tool.reference_bank.requires_grad
    assert tool.reference_bank.grad_fn is None
    assert tool.score(torch.empty((0, 2))).shape == (0,)


def test_tool_validates_fit_and_query_contract():
    tool = GoldNearestNeighborNoveltyScoringTool()
    with pytest.raises(ValueError, match="Fit the novelty scorer"):
        tool.score(torch.ones((1, 2)))
    with pytest.raises(ValueError, match="must not be empty"):
        tool.fit(torch.empty((0, 2)))
    with pytest.raises(ValueError, match="2D tensors"):
        tool.fit(torch.ones(2))
    with pytest.raises(ValueError, match="finite"):
        tool.fit(torch.tensor([[float("nan")]]))
    tool.fit(torch.ones((1, 2)))
    with pytest.raises(ValueError, match="feature_dim"):
        tool.score(torch.ones((1, 3)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk": 0},
        {"chunk": True},
        {"batch_size": 0},
        {"num_workers": -1},
        {"max_batches": -1},
        {"score_key": "idx"},
        {"vectorized_key": "idx_vector"},
        {"score_key": "features", "vectorized_key": "features"},
        {"vectorized_key": "features", "to_keep_schema": {"features": pxt.Float}},
        {"to_keep_schema": {"idx": pxt.Int}},
    ],
)
def test_scorer_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GoldNoveltyScorer(table_path="unused.output", **kwargs)


class VectorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class CountingNearestScorer(GoldNearestNeighborNoveltyScoringTool):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def score(self, x):
        self.calls += 1
        return super().score(x)


def test_table_scoring_resumes_pending_rows_and_preserves_vector_ids(clean_unit_test):
    reference = pxt.create_table(
        "unit_test.novelty_reference",
        source=[
            {
                "idx": 10,
                "idx_vector": 50,
                "features": np.array([1.0, 0.0], dtype=np.float32),
            }
        ],
        if_exists="replace_force",
    )
    source = pxt.create_table(
        "unit_test.novelty_source",
        source=[
            {
                "idx": 1,
                "idx_vector": 100,
                "features": np.array([1.0, 0.0], dtype=np.float32),
                "attrs": np.array([1, 2]),
            },
            {
                "idx": 1,
                "idx_vector": 105,
                "features": np.array([0.0, 1.0], dtype=np.float32),
                "attrs": np.array([3, 4]),
            },
            {
                "idx": 2,
                "idx_vector": 111,
                "features": np.array([-1.0, 0.0], dtype=np.float32),
                "attrs": np.array([5, 6]),
            },
        ],
        if_exists="replace_force",
    )
    tensor_scorer = CountingNearestScorer()
    scorer = GoldNoveltyScorer(
        table_path="unit_test.novelty_output",
        scorer=tensor_scorer,
        vectorized_key="features",
        score_key="novelty_score",
        include_vectorized_in_table=True,
        to_keep_schema={"attrs": pxt.Array[(2,), pxt.Int]},
        chunk=1,
        max_batches=1,
    )

    result = scorer.score_in_table(source, reference)
    assert GoldNoveltyScorer.get_novelty_scores(result, "novelty_score") == {100: 0.0}
    assert tensor_scorer.calls == 1
    restarted_tool = CountingNearestScorer()
    restarted_scorer = GoldNoveltyScorer(
        table_path="unit_test.novelty_output",
        scorer=restarted_tool,
        vectorized_key="features",
        score_key="novelty_score",
        include_vectorized_in_table=True,
        to_keep_schema={"attrs": pxt.Array[(2,), pxt.Int]},
        chunk=1,
        max_batches=1,
    )
    result = restarted_scorer.score_in_table(source, reference)
    assert restarted_tool.calls == 1
    result = restarted_scorer.score_in_table(source)
    assert restarted_tool.calls == 2
    assert GoldNoveltyScorer.get_novelty_scores(
        result, "novelty_score"
    ) == pytest.approx({100: 0.0, 105: 1.0, 111: 2.0})
    result = restarted_scorer.score_in_table(source)
    assert restarted_tool.calls == 2
    assert GoldNoveltyScorer.get_novel_indices(
        result, 1.0, score_key="novelty_score"
    ) == {2}
    assert GoldNoveltyScorer.get_novel_indices(
        result, 1.0, idx_key="idx_vector", score_key="novelty_score"
    ) == {111}
    assert result.count() == 3
    rows = result.select(result.idx_vector, result.features, result.attrs).collect()
    assert {row["idx_vector"] for row in rows} == {100, 105, 111}
    assert all(row["features"].shape == (2,) for row in rows)
    assert {tuple(row["attrs"]) for row in rows} == {(1, 2), (3, 4), (5, 6)}

    source.insert(
        [
            {
                "idx": 3,
                "idx_vector": 200,
                "features": np.array([1.0, 1.0], dtype=np.float32),
                "attrs": np.array([7, 8]),
            }
        ]
    )
    result = restarted_scorer.score_in_table(source)
    assert restarted_tool.calls == 3
    assert result.count() == 4
    restarted_scorer.score_in_table(source)
    assert restarted_tool.calls == 3
    result.batch_update(
        [
            {
                "idx": 9,
                "idx_vector": 999,
                "novelty_score": None,
                "features": np.array([0.5, 0.5], dtype=np.float32),
                "attrs": np.array([9, 9]),
            }
        ],
        if_not_exists="insert",
    )
    with pytest.raises(ValueError, match="missing from the supplied source"):
        restarted_scorer.score_in_table(source)
    with pytest.raises(ValueError, match="allow_existing"):
        GoldNoveltyScorer(
            table_path="unit_test.novelty_output", allow_existing=False
        ).score_in_table(source, reference)


class OffsetReducer(GoldReductionToolWithFit):
    def __init__(self):
        self.fit_calls = 0
        self.offset = None

    def fit(self, x):
        self.fit_calls += 1
        self.offset = x.mean(dim=0)

    def transform(self, x):
        return x - self.offset

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def test_reducer_is_fitted_once_on_reference_and_used_for_query_chunks(
    clean_unit_test,
):
    reducer = OffsetReducer()
    reference = VectorDataset(
        [
            {"vectorized": torch.tensor([0.0, 0.0]), "idx": 0},
            {"vectorized": torch.tensor([2.0, 0.0]), "idx": 1},
        ]
    )
    queries = VectorDataset(
        [
            {"vectorized": torch.tensor([1.0, 0.0]), "idx": 2},
            {"vectorized": torch.tensor([3.0, 0.0]), "idx": 3},
        ]
    )
    scorer = GoldNoveltyScorer(
        table_path="unit_test.novelty_reduced",
        reducer=reducer,
        chunk=1,
    )

    result = scorer.score_in_table(queries, reference)

    assert reducer.fit_calls == 1
    assert reducer.offset.tolist() == [1.0, 0.0]
    assert GoldNoveltyScorer.get_novelty_scores(result) == pytest.approx(
        {0: 1.0, 1: 0.0}
    )
    with pytest.raises(ValueError, match="Cannot refit"):
        scorer.fit_scorer(reference)
    assert reducer.fit_calls == 1


def test_reducer_without_transform_is_rejected_before_fitting(clean_unit_test):
    from sklearn.manifold import TSNE

    from goldener.reduce import GoldSKLearnReductionTool

    scorer = GoldNoveltyScorer(
        table_path="unit_test.novelty_tsne",
        reducer=GoldSKLearnReductionTool(TSNE(n_components=2, perplexity=2)),
    )
    reference = VectorDataset([{"vectorized": torch.tensor([1.0, 0.0]), "idx": 0}])

    with pytest.raises(ValueError, match="supports transform"):
        scorer.fit_scorer(reference)
    scorer.scorer.fit(torch.ones((1, 2)))
    with pytest.raises(ValueError, match="supports transform"):
        scorer.score_in_table(reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dataset_scoring_uses_index_fallbacks_and_drop_table_snapshot(
    clean_unit_test, dtype
):
    reference = VectorDataset(
        [{"vectorized": torch.tensor([1.0, 0.0], dtype=dtype), "idx": 8}]
    )
    queries = VectorDataset(
        [
            {"vectorized": torch.tensor([1.0, 0.0], dtype=dtype)},
            {"vectorized": torch.tensor([0.0, 1.0], dtype=dtype)},
        ]
    )
    scorer = GoldNoveltyScorer(
        table_path="unit_test.novelty_dataset", batch_size=2, drop_table=True
    )

    result = scorer.score_in_dataset(queries, reference)

    assert len(result) == 2
    samples = list(result)
    assert [sample["idx_vector"] for sample in samples] == [0, 1]
    assert [sample["novelty"] for sample in samples] == pytest.approx([0.0, 1.0])
    if dtype is torch.bfloat16:
        assert all(
            np.asarray(sample["vectorized"]).dtype == np.float32 for sample in samples
        )
        np.testing.assert_array_equal(
            np.asarray(samples[0]["vectorized"]), np.array([1.0, 0.0], dtype=np.float32)
        )
    with pytest.raises(Error):
        pxt.get_table("unit_test.novelty_dataset")


def test_table_scoring_can_omit_original_vectors(clean_unit_test):
    source = pxt.create_table(
        "unit_test.novelty_source_without_vectors",
        source=[
            {
                "idx": 4,
                "idx_vector": 90,
                "vectorized": np.array([1.0, 0.0], dtype=np.float32),
            },
            {
                "idx": 4,
                "idx_vector": 95,
                "vectorized": np.array([0.0, 1.0], dtype=np.float32),
            },
        ],
        if_exists="replace_force",
    )
    reference = VectorDataset([{"vectorized": torch.tensor([1.0, 0.0]), "idx": 0}])
    scorer = GoldNoveltyScorer(table_path="unit_test.novelty_scores_only")

    result = scorer.score_in_table(source, reference)

    assert "vectorized" not in result.columns()
    assert GoldNoveltyScorer.get_novelty_scores(result) == pytest.approx(
        {90: 0.0, 95: 1.0}
    )


def test_existing_scalar_vector_column_is_rejected(clean_unit_test):
    source = pxt.create_table(
        "unit_test.novelty_vector_schema_source",
        source=[
            {
                "idx": 0,
                "idx_vector": 1,
                "vectorized": np.array([1.0, 0.0], dtype=np.float32),
            }
        ],
        if_exists="replace_force",
    )
    pxt.create_table(
        "unit_test.novelty_vector_schema_output",
        schema={
            "idx": pxt.Required[pxt.Int],
            "idx_vector": pxt.Required[pxt.Int],
            "vectorized": pxt.Float,
            "novelty": pxt.Float,
        },
        primary_key="idx_vector",
        if_exists="replace_force",
    )
    scorer = GoldNoveltyScorer(
        table_path="unit_test.novelty_vector_schema_output",
        include_vectorized_in_table=True,
    )
    reference = VectorDataset([{"vectorized": torch.tensor([1.0, 0.0]), "idx": 0}])

    with pytest.raises(ValueError, match="vector column has an incompatible shape"):
        scorer.score_in_table(source, reference)


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ({"vectorized": torch.tensor(1.0)}, "shape"),
        ({"vectorized": torch.tensor([1, 2])}, "floating-point"),
        ({"vectorized": torch.tensor([float("inf"), 0.0])}, "finite"),
        (
            {"vectorized": torch.tensor([1.0, 0.0]), "idx_vector": 1.5},
            "integer scalars",
        ),
    ],
)
def test_source_vector_validation(sample, message):
    scorer = GoldNoveltyScorer(table_path="unused.output")
    with pytest.raises(ValueError, match=message):
        scorer._vectors(VectorDataset([sample]))


def test_source_rejects_duplicate_vector_ids():
    scorer = GoldNoveltyScorer(table_path="unused.output")
    source = VectorDataset(
        [
            {"vectorized": torch.tensor([1.0, 0.0]), "idx_vector": 5},
            {"vectorized": torch.tensor([0.0, 1.0]), "idx_vector": 5},
        ]
    )

    with pytest.raises(ValueError, match="idx_vector values must be unique"):
        scorer._vectors(source)


@pytest.mark.parametrize(
    "reduced",
    [None, [1.0, 2.0], torch.ones((2, 2)), torch.tensor([[float("nan")]])],
    ids=["none", "list", "wrong-row-count", "nonfinite"],
)
def test_invalid_reducer_output_is_rejected(reduced):
    with pytest.raises(ValueError, match="Reducer output"):
        GoldNoveltyScorer._validate_reduced(reduced, expected_rows=1)


class InvalidOutputScorer(GoldNearestNeighborNoveltyScoringTool):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.reference_bank = torch.ones((1, 2))

    def score(self, x):
        match self.output:
            case "none":
                return None
            case "list":
                return [0.0] * len(x)
            case "wrong-shape":
                return torch.zeros((len(x), 1))
            case "nonfinite":
                return torch.full((len(x),), float("nan"))


@pytest.mark.parametrize("output", ["none", "list", "wrong-shape", "nonfinite"])
def test_invalid_scorer_output_is_rejected(clean_unit_test, output):
    source = pxt.create_table(
        "unit_test.invalid_score_source",
        source=[
            {
                "idx": 1,
                "idx_vector": 10,
                "vectorized": np.array([1.0, 0.0], dtype=np.float32),
            }
        ],
        if_exists="replace_force",
    )
    scorer = GoldNoveltyScorer(
        table_path="unit_test.invalid_score_output",
        scorer=InvalidOutputScorer(output),
    )

    with pytest.raises(ValueError, match="one finite score per query"):
        scorer.score_in_table(source)
