from statistics import mean
from unittest.mock import Mock

import pytest
import torch
import numpy as np
import pixeltable as pxt
from pixeltable import Error
from torch.utils.data import Dataset

from goldener.distance import DistanceType
from goldener.novelty import (
    GoldNearestNeighborNoveltyScoringTool,
    GoldNoveltyScorer,
)
from goldener.reduce import GoldReductionTool, GoldReductionToolWithFit


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


@pytest.fixture
def pxt_unit_test_dir():
    pxt.drop_dir("unit_test", force=True)
    pxt.create_dir("unit_test", if_exists="ignore")
    yield
    pxt.drop_dir("unit_test", force=True)


class TestGoldNearestNeighborNoveltyScoringTool:
    @pytest.mark.parametrize(
        ("distance", "expected"),
        [
            (DistanceType.COSINE, [0.0, 1.0, 1.0]),
            (DistanceType.EUCLIDEAN, [3.0, 1.0, 0.0]),
        ],
    )
    def test_nearest_reference_scores_and_blocks(self, distance, expected):
        reference = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        queries = torch.tensor([[4.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        tool = GoldNearestNeighborNoveltyScoringTool(distance, distance_batch_size=1)
        tool.fit(reference)

        scores = tool.score(queries)

        assert scores.tolist() == pytest.approx(expected)
        unblocked = GoldNearestNeighborNoveltyScoringTool(
            distance, distance_batch_size=8
        )
        unblocked.fit(reference)
        assert torch.equal(unblocked.score(queries), scores)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_score_restores_query_dtype(self, dtype):
        tool = GoldNearestNeighborNoveltyScoringTool()
        tool.fit(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        query = torch.tensor([[0.0, 1.0]], dtype=dtype)

        result = tool.score(query)

        assert result.dtype == dtype
        assert result.device == query.device
        assert result.item() == pytest.approx(1.0)

    def test_fit_keeps_an_independent_reference(self):
        reference = torch.tensor([[1.0, 2.0]])
        reference.requires_grad_()
        tool = GoldNearestNeighborNoveltyScoringTool()
        tool.fit(reference)
        with torch.no_grad():
            reference.zero_()

        scores = tool.score(torch.tensor([[1.0, 2.0]]))
        assert scores.tolist() == pytest.approx([0.0], abs=1e-6)

    def test_tool_validates_fit_and_query_contract(self):
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
        with pytest.raises(ValueError, match="query must not be empty"):
            tool.score(torch.empty((0, 2)))
        with pytest.raises(ValueError, match="feature_dim"):
            tool.score(torch.ones((1, 3)))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"chunk": 0}, "chunk", id="zero-chunk"),
        pytest.param({"chunk": True}, "chunk", id="boolean-chunk"),
        pytest.param({"batch_size": 0}, "batch_size", id="zero-batch-size"),
        pytest.param({"num_workers": -1}, "num_workers", id="negative-workers"),
        pytest.param({"max_batches": -1}, "max_batches", id="negative-batch-limit"),
        pytest.param(
            {"to_keep_schema": {"vectorized": pxt.Float}},
            "to_keep_schema",
            id="retained-vector-column",
        ),
    ],
)
def test_scorer_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GoldNoveltyScorer(
            scorer=GoldNearestNeighborNoveltyScoringTool(),
            table_path="unused.output",
            **kwargs,
        )


@pytest.mark.usefixtures("pxt_unit_test_dir")
class TestGoldNoveltyScorer:
    def test_table_scoring_resumes_pending_rows_and_preserves_vector_ids(
        self, monkeypatch
    ):
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
        tensor_scorer.fit(torch.tensor([[1.0, 0.0]]))
        options = dict(
            vectorized_key="features",
            score_key="novelty_score",
            include_vectorized_in_table=True,
            to_keep_schema={"attrs": pxt.Array[(2,), pxt.Int]},
            chunk=1,
        )
        scorer = GoldNoveltyScorer(
            "unit_test.novelty_output",
            scorer=tensor_scorer,
            max_batches=1,
            **options,
        )

        result = scorer.score_in_table(source)
        assert GoldNoveltyScorer.get_novelty_scores_per_idx(
            result, "novelty_score"
        ) == {100: 0.0}
        assert tensor_scorer.calls == 1
        restarted_tool = CountingNearestScorer()
        restarted_tool.fit(torch.tensor([[1.0, 0.0]]))
        restarted_scorer = GoldNoveltyScorer(
            scorer.table_path,
            scorer=restarted_tool,
            **options,
        )
        score = restarted_tool.score

        def interrupt_after_one_chunk(vectors):
            if restarted_tool.calls == 1:
                raise RuntimeError("interrupted")
            return score(vectors)

        with monkeypatch.context() as patch:
            patch.setattr(restarted_tool, "score", interrupt_after_one_chunk)
            with pytest.raises(RuntimeError, match="interrupted"):
                restarted_scorer.score_in_table(source)
        assert restarted_tool.calls == 1
        partially_scored = GoldNoveltyScorer.get_novelty_scores_per_idx(
            result, "novelty_score"
        )
        assert len(partially_scored) == 2
        assert partially_scored[100] == 0.0

        result = restarted_scorer.score_in_table(source)
        assert restarted_tool.calls == 2
        assert GoldNoveltyScorer.get_novelty_scores_per_idx(
            result, "novelty_score"
        ) == pytest.approx({100: 0.0, 105: 1.0, 111: 2.0})
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
        result = restarted_scorer.score_in_table(source)
        assert 999 in GoldNoveltyScorer.get_novelty_scores_per_idx(
            result, "novelty_score"
        )
        assert restarted_tool.calls == 4
        restarted_scorer.score_in_table(source)
        assert restarted_tool.calls == 4
        restarted_scorer.allow_existing = False
        with pytest.raises(ValueError, match="allow_existing"):
            restarted_scorer.score_in_table(source)

    def test_external_tool_state_is_preserved_when_scoring_query_chunks(self):
        reducer = OffsetReducer()
        vectors = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        reducer.fit(vectors)
        tool = GoldNearestNeighborNoveltyScoringTool()
        tool.fit(reducer.transform(vectors))
        queries = VectorDataset(
            [
                {"vectorized": torch.tensor([1.0, 0.0]), "idx": 2},
                {"vectorized": torch.tensor([3.0, 0.0]), "idx": 3},
            ]
        )
        scorer = GoldNoveltyScorer(
            table_path="unit_test.novelty_reduced",
            scorer=tool,
            reducer=reducer,
            chunk=1,
        )

        result = scorer.score_in_table(queries)

        assert reducer.fit_calls == 1
        assert reducer.offset.tolist() == [1.0, 0.0]
        assert GoldNoveltyScorer.get_novelty_scores_per_idx(result) == pytest.approx(
            {0: 1.0, 1: 0.0}
        )
        scorer.score_in_table(queries)
        assert reducer.fit_calls == 1

    @pytest.mark.parametrize(
        "dtype, batch_size, insert_size",
        [
            pytest.param(torch.float32, 1, 1, id="float32-immediate-inserts"),
            pytest.param(torch.bfloat16, 2, 100, id="bfloat16-buffered-insert"),
        ],
    )
    def test_dataset_scoring_uses_index_fallbacks_and_drop_table_snapshot(
        self, dtype, batch_size, insert_size
    ):
        tool = GoldNearestNeighborNoveltyScoringTool()
        tool.fit(torch.tensor([[1.0, 0.0]], dtype=dtype))
        queries = VectorDataset(
            [
                {
                    "vectorized": torch.tensor([1.0, 0.0], dtype=dtype),
                    "labels": ["cat", "dog"],
                    "attrs": torch.tensor([3.0]),
                },
                {
                    "vectorized": torch.tensor([0.0, 1.0], dtype=dtype),
                    "labels": ["bird"],
                    "attrs": torch.tensor([4.0]),
                },
            ]
        )
        scorer = GoldNoveltyScorer(
            table_path="unit_test.novelty_dataset",
            scorer=tool,
            batch_size=batch_size,
            min_pxt_insert_size=insert_size,
            to_keep_schema={"labels": pxt.Json, "attrs": pxt.Array[(1,), pxt.Float]},
            drop_table=True,
        )

        result = scorer.score_in_dataset(queries)

        assert len(result) == 2
        samples = list(result)
        assert [sample["idx"] for sample in samples] == [0, 1]
        assert [sample["idx_vector"] for sample in samples] == [0, 1]
        assert [sample["novelty"] for sample in samples] == pytest.approx([0.0, 1.0])
        assert [sample["labels"] for sample in samples] == [["cat", "dog"], ["bird"]]
        assert [sample["attrs"].shape for sample in samples] == [(1,), (1,)]
        assert [sample["attrs"].item() for sample in samples] == [3.0, 4.0]
        assert all(
            np.asarray(sample["vectorized"]).dtype == np.float32 for sample in samples
        )
        np.testing.assert_array_equal(
            np.stack([sample["vectorized"] for sample in samples]), np.eye(2)
        )
        with pytest.raises(Error):
            pxt.get_table("unit_test.novelty_dataset")

    def test_table_scoring_can_omit_original_vectors(self):
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
        tool = GoldNearestNeighborNoveltyScoringTool()
        tool.fit(torch.tensor([[1.0, 0.0]]))
        scorer = GoldNoveltyScorer(
            table_path="unit_test.novelty_scores_only", scorer=tool, max_batches=0
        )

        result = scorer.score_in_table(source)
        assert result.count() == 2
        assert GoldNoveltyScorer.get_novelty_scores_per_idx(result) == {}
        scorer.max_batches = None
        result = scorer.score_in_table(source)

        assert "vectorized" not in result.columns()
        assert GoldNoveltyScorer.get_novelty_scores_per_idx(result) == pytest.approx(
            {90: 0.0, 95: 1.0}
        )
        result.insert([{"idx": 9, "idx_vector": 999, "novelty": None}])
        with pytest.raises(ValueError, match="missing from the supplied source"):
            scorer.score_in_table(source)

    def test_existing_scalar_vector_column_is_rejected(self):
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
            scorer=GoldNearestNeighborNoveltyScoringTool(),
            table_path="unit_test.novelty_vector_schema_output",
            include_vectorized_in_table=True,
        )

        with pytest.raises(ValueError, match="vector column has an incompatible shape"):
            scorer.setup_scoring_table(source)

    @pytest.mark.parametrize(
        ("samples", "match"),
        [
            pytest.param(
                [{"vectorized": torch.tensor(1.0)}], "shape", id="scalar-vector"
            ),
            pytest.param(
                [{"vectorized": torch.tensor([1, 2])}], "floating", id="integer-vector"
            ),
            pytest.param(
                [{"vectorized": torch.tensor([float("inf"), 0.0])}],
                "finite",
                id="nonfinite-vector",
            ),
            pytest.param(
                [{"vectorized": torch.tensor([1.0, 0.0]), "idx_vector": 1.5}],
                "integer",
                id="fractional-vector-id",
            ),
            pytest.param(
                [
                    {"vectorized": torch.tensor([1.0, 0.0]), "idx_vector": 5},
                    {"vectorized": torch.tensor([0.0, 1.0]), "idx_vector": 5},
                ],
                "idx_vector.*unique",
                id="duplicate-vector-id",
            ),
        ],
    )
    def test_rejects_invalid_dataset_rows(self, samples, match):
        scorer = GoldNoveltyScorer(
            scorer=GoldNearestNeighborNoveltyScoringTool(),
            table_path="unit_test.invalid_source",
        )
        with pytest.raises(ValueError, match=match):
            scorer.setup_scoring_table(VectorDataset(samples))

    @pytest.mark.parametrize(
        "supports_transform, output, match",
        [
            pytest.param(False, None, "supports transform", id="unsupported-transform"),
            pytest.param(True, torch.ones((2, 2)), None, id="changed-vector-count"),
        ],
    )
    def test_invalid_reducer_is_rejected_before_scoring(
        self, supports_transform, output, match
    ):
        reducer = Mock(spec=GoldReductionTool, supports_transform=supports_transform)
        reducer.transform.return_value = output
        tool = CountingNearestScorer()
        tool.fit(torch.ones((1, 2)))
        scorer = GoldNoveltyScorer(
            "unit_test.invalid_reducer",
            scorer=tool,
            reducer=reducer,
        )
        source = VectorDataset([{"vectorized": torch.ones(2)}])

        with pytest.raises(ValueError, match=match):
            scorer.score_in_table(source)

        assert tool.calls == 0

    @pytest.mark.parametrize(
        "output, match",
        [
            pytest.param(torch.zeros(2), "one entry per row", id="wrong-score-count"),
            pytest.param(torch.tensor([float("nan")]), "finite", id="nonfinite-score"),
        ],
    )
    def test_invalid_scorer_output_leaves_row_pending(self, output, match):
        tool = GoldNearestNeighborNoveltyScoringTool()
        tool.fit(torch.ones((1, 2)))
        tool.score = Mock(return_value=output)
        scorer = GoldNoveltyScorer(
            "unit_test.invalid_score_output",
            scorer=tool,
        )
        source = VectorDataset([{"vectorized": torch.ones(2)}])

        with pytest.raises(ValueError, match=match):
            scorer.score_in_table(source)

        table = pxt.get_table(scorer.table_path)
        assert table.count() == 1
        assert table.collect()[0]["novelty"] is None

    @pytest.mark.parametrize(
        "source_kind, per_label",
        [
            pytest.param("dataset", False, id="dataset-shared-tool"),
            pytest.param("table", True, id="table-per-label-tools"),
        ],
    )
    def test_scores_labels_with_shared_or_separate_tools(self, source_kind, per_label):
        samples = [
            {
                "idx": 9,
                "idx_vector": 20,
                "vectorized": np.array([1.0, 0.0], dtype=np.float32),
                "label": "a",
            },
            {
                "idx": 9,
                "idx_vector": 30,
                "vectorized": np.array([1.0, 0.0], dtype=np.float32),
                "label": "all_labels",
            },
        ]
        source = (
            pxt.create_table("unit_test.label_source", source=samples)
            if source_kind == "table"
            else VectorDataset(
                [
                    {**sample, "vectorized": torch.from_numpy(sample["vectorized"])}
                    for sample in samples
                ]
            )
        )
        first = CountingNearestScorer()
        first.fit(torch.tensor([[1.0, 0.0]]))
        second = CountingNearestScorer()
        second.fit(torch.tensor([[0.0, 1.0]]))
        scorer = GoldNoveltyScorer(
            table_path="unit_test.label_scores",
            scorer={"a": first, "all_labels": second} if per_label else first,
            label_key="label",
            chunk=1,
            batch_size=2,
            max_batches=1 if per_label else None,
        )
        if per_label:
            del scorer.scorer["all_labels"]
            with pytest.raises(ValueError, match="No scorer for labels.*all_labels"):
                scorer.score_in_table(source)
            assert first.calls == 0
            scorer.scorer["all_labels"] = second

        result = scorer.score_in_table(source)
        if per_label:
            assert scorer.get_novelty_scores_per_idx(result) == {20: 0.0}
            assert second.calls == 0
            result = scorer.score_in_table(source)

        assert scorer.get_novelty_scores_per_idx(result) == pytest.approx(
            {20: 0.0, 30: 1.0 if per_label else 0.0}
        )
        assert first.calls == (1 if per_label else 2)
        assert second.calls == (1 if per_label else 0)
        assert {row["label"] for row in result.select(result.label).collect()} == {
            "a",
            "all_labels",
        }

    @pytest.mark.parametrize(
        "options, expected",
        [
            pytest.param(
                {"threshold": 0.0, "k": 3},
                [(111, 2.0), (105, 1.0)],
                id="highest-vectors-above-threshold",
            ),
            pytest.param(
                {"threshold": 2.0, "k": 3, "best": False},
                [(100, 0.0), (105, 1.0)],
                id="lowest-vectors-below-threshold",
            ),
            pytest.param(
                {"k": 1},
                [(111, 2.0)],
                id="limit-results",
            ),
            pytest.param(
                {"idx_key": "idx", "k": 2},
                [(2, 2.0), (1, 1.0)],
                id="highest-samples-use-max",
            ),
            pytest.param(
                {
                    "idx_key": "idx",
                    "aggregation": mean,
                    "threshold": 1.0,
                    "best": False,
                },
                [(1, 0.5)],
                id="aggregate-before-filtering",
            ),
            pytest.param(
                {"idx_key": "idx", "label_key": "label", "label_value": "a", "k": 2},
                [(2, 2.0), (1, 0.0)],
                id="filter-label-before-aggregating",
            ),
            pytest.param({"k": 0}, [], id="zero-limit"),
            pytest.param(
                {"label_key": "label", "label_value": "absent"}, [], id="missing-label"
            ),
        ],
    )
    def test_retrieve_novelty_scores(self, options, expected):
        table = pxt.create_table(
            "unit_test.retrieval_scores",
            schema={
                "idx": pxt.Int,
                "idx_vector": pxt.Int,
                "novelty_score": pxt.Float,
                "label": pxt.String,
            },
        )
        table.insert(
            [
                {"idx": 1, "idx_vector": 100, "novelty_score": 0.0, "label": "a"},
                {"idx": 1, "idx_vector": 105, "novelty_score": 1.0, "label": "b"},
                {"idx": 2, "idx_vector": 111, "novelty_score": 2.0, "label": "a"},
                {"idx": 3, "idx_vector": 120, "novelty_score": None, "label": "a"},
            ]
        )
        scores = GoldNoveltyScorer.get_novelty_scores_per_idx(
            table, "novelty_score", **options
        )

        assert list(scores.items()) == expected


def test_retrieval_rejects_negative_limit():
    with pytest.raises(ValueError, match="k must be a nonnegative integer"):
        GoldNoveltyScorer.get_novelty_scores_per_idx(Mock(spec=pxt.Table), k=-1)
