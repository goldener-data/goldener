from unittest.mock import Mock

import pytest
import pixeltable as pxt
import torch
from torch.utils.data import Dataset

from goldener.clusterize import get_random_chunk_assignment
from goldener.reduce import GoldReductionTool, GoldReductionToolWithFit
from goldener.distance import DistanceType
from goldener.torch_utils import collate_keeping_sequences_as_sequences
from goldener.weigh import (
    GoldDensityWeigher,
    GoldDensityWeighingTool,
    GoldKnnDensityWeighingTool,
)


class VectorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample.copy() if isinstance(sample, dict) else sample


def custom_collate(batch):
    return collate_keeping_sequences_as_sequences(batch)


class CountingWeigher(GoldDensityWeighingTool):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.batch_shapes = []

    def weigh_batch(self, x, batch, sample_idx=None, batch_sample_idx=None):
        self.batch_shapes.append((len(x), len(batch)))
        return self.weigh(batch, sample_idx=sample_idx)

    def weigh(self, x, sample_idx=None):
        self.calls += 1
        return x.new_ones((len(x),))


class CoordinateWeigher(CountingWeigher):
    def weigh(self, x, sample_idx=None):
        self.calls += 1
        return x[:, 0] + 1


class TrackingReducer(GoldReductionTool):
    def __init__(self):
        self.inputs = []

    def transform(self, x):
        self.inputs.append(x.clone())
        return x[:, :1] * 2


class FittedTrackingReducer(TrackingReducer, GoldReductionToolWithFit):
    def __init__(self):
        super().__init__()
        self.fitted_inputs = []

    def fit(self, x):
        self.fitted_inputs.append(x.clone())

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class FixedScoreWeigher(GoldDensityWeighingTool):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def weigh_batch(self, x, batch, sample_idx=None, batch_sample_idx=None):
        scores = self.weigh(x, sample_idx=sample_idx)
        if scores.shape != (len(x),):
            return scores
        indices = (batch[:, None, :] == x[None, :, :]).all(dim=2).long().argmax(dim=1)
        return scores[indices]

    def weigh(self, x, sample_idx=None):
        if isinstance(self.values, Exception):
            raise self.values
        return torch.tensor(self.values, dtype=torch.float64)


class InterruptingWeigher(CoordinateWeigher):
    def __init__(self, interrupt):
        super().__init__()
        self.interrupt = interrupt
        self.calls = 0

    def weigh(self, x, sample_idx=None):
        self.calls += 1
        if self.interrupt and self.calls == 2:
            raise RuntimeError("simulated interruption")
        return x[:, 0] + 1


@pytest.fixture
def pxt_unit_test_dir():
    pxt.drop_dir("unit_test", force=True)
    pxt.create_dir("unit_test", if_exists="ignore")
    yield
    pxt.drop_dir("unit_test", force=True)


class TestGoldKnnDensityWeighingTool:
    @pytest.mark.parametrize(
        "k,expected_distances",
        [(1, [0.1, 0.1, 0.1, 4.8]), (2, [0.15, 0.1, 0.15, 4.85])],
    )
    def test_knn_density_returns_raw_distance_power_and_sparse_points_score_higher(
        self,
        k,
        expected_distances,
    ):
        x = torch.tensor([[0.0], [0.1], [0.2], [5.0]])
        scores = GoldKnnDensityWeighingTool(k=k).weigh(x)

        assert scores.shape == (4,)
        assert scores.dtype == x.dtype
        assert scores.device == x.device
        torch.testing.assert_close(scores, torch.tensor(expected_distances).sqrt())
        assert scores[3] > scores[0]

    @pytest.mark.parametrize("k", [1, 5])
    @pytest.mark.parametrize("power", [0, 0.5])
    @pytest.mark.parametrize(
        "size_offset", [0, 1, 2], ids=["empty", "singleton", "k-plus-one"]
    )
    def test_knn_density_rejects_insufficient_populations(self, k, power, size_offset):
        size = k + 1 if size_offset == 2 else size_offset
        with pytest.raises(ValueError, match=r"at least k \+ 2"):
            GoldKnnDensityWeighingTool(k=k, power=power).weigh(torch.zeros(size, 2))

    @pytest.mark.parametrize("k", [1, 5])
    def test_knn_density_zero_power_at_minimum_population(self, k):
        scores = GoldKnnDensityWeighingTool(k=k, power=0).weigh(torch.zeros(k + 2, 2))
        assert torch.equal(scores, torch.ones(k + 2))

    def test_knn_density_uses_absolute_eps_for_identical_vectors(self):
        scores = GoldKnnDensityWeighingTool(power=0.5, eps=1e-8).weigh(
            torch.zeros((7, 2))
        )

        assert torch.equal(scores, torch.full((7,), 1e-4))

    @pytest.mark.parametrize(
        "row_indices",
        [[3, 0, 2, 1], [2], [1, 1, 3], []],
        ids=["reordered", "single-query", "duplicate-queries", "empty-query"],
    )
    def test_knn_density_excludes_only_self_and_query_batching_preserves_results(
        self,
        row_indices,
    ):
        x = torch.tensor([[0.0], [0.0], [3.0], [8.0]])
        one_batch = GoldKnnDensityWeighingTool(k=1, distance_batch_size=20).weigh(x)
        small_batches = GoldKnnDensityWeighingTool(k=1, distance_batch_size=1).weigh(x)

        indices = torch.tensor(row_indices, dtype=torch.long)
        tensor_scores = GoldKnnDensityWeighingTool(k=1).weigh_batch(x, x[indices])
        assert torch.allclose(tensor_scores, one_batch[indices])
        assert tensor_scores.dtype == x.dtype
        assert tensor_scores.device == x.device
        assert torch.allclose(one_batch, small_batches)
        assert one_batch[0] == one_batch[1]
        assert one_batch[0].item() == pytest.approx(1e-6)
        assert one_batch[2] > one_batch[0]

    def test_knn_density_preserves_small_distances_in_large_dynamic_range(self):
        x = torch.tensor([[0.0], [1.0], [1e30]], dtype=torch.float32)

        scores = GoldKnnDensityWeighingTool(k=1).weigh(x)

        assert torch.allclose(scores, torch.tensor([1.0, 1.0, 1e15]))

    def test_knn_raw_scores_scale_in_original_distance_units(self):
        x = torch.tensor([[0.0], [1.0], [3.0]])
        tool = GoldKnnDensityWeighingTool(k=1)

        scores = tool.weigh(x)
        scaled_scores = tool.weigh(x * 10)

        assert torch.allclose(scaled_scores, scores * (10**0.5))

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            pytest.param({"k": 0}, "k must be a positive integer", id="zero-k"),
            pytest.param({"k": True}, "k must be a positive integer", id="boolean-k"),
            pytest.param(
                {"distance_batch_size": 0},
                "distance_batch_size must be a positive integer",
                id="zero-distance-batch-size",
            ),
            pytest.param(
                {"power": -1},
                "power must be finite and nonnegative",
                id="negative-power",
            ),
            pytest.param(
                {"power": float("nan")},
                "power must be finite and nonnegative",
                id="nan-power",
            ),
            pytest.param(
                {"power": float("inf")},
                "power must be finite and nonnegative",
                id="infinite-power",
            ),
            pytest.param(
                {"power": True},
                "power must be finite and nonnegative",
                id="boolean-power",
            ),
            pytest.param({"eps": 0}, "eps must be finite and positive", id="zero-eps"),
            pytest.param(
                {"eps": float("nan")}, "eps must be finite and positive", id="nan-eps"
            ),
            pytest.param(
                {"eps": float("inf")},
                "eps must be finite and positive",
                id="infinite-eps",
            ),
            pytest.param(
                {"eps": True}, "eps must be finite and positive", id="boolean-eps"
            ),
        ],
    )
    def test_knn_density_rejects_invalid_configuration(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GoldKnnDensityWeighingTool(**kwargs)

    @pytest.mark.parametrize(
        "x,match",
        [
            pytest.param(
                torch.ones(3), "only accepts 2D tensors", id="one-dimensional"
            ),
            pytest.param(
                torch.empty((2, 0)), "positive feature width", id="empty-features"
            ),
            pytest.param(
                torch.tensor([[float("nan")]]),
                "finite floating point values",
                id="nonfinite-input",
            ),
            pytest.param(
                torch.ones((7, 2), dtype=torch.int64),
                "finite floating point values",
                id="integer-input",
            ),
        ],
    )
    def test_knn_density_rejects_invalid_input(self, x, match):
        with pytest.raises(ValueError, match=match):
            GoldKnnDensityWeighingTool().weigh(x)
        with pytest.raises(ValueError, match=match):
            GoldKnnDensityWeighingTool().weigh_batch(torch.zeros(7, 2), x)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16, torch.float32, torch.float64],
        ids=["float16", "bfloat16", "float32", "float64"],
    )
    def test_knn_density_preserves_supported_float_dtypes(self, dtype):
        x = torch.tensor([[0.0, 1.0], [0.1, 1.0], [4.0, 1.0]], dtype=dtype)

        scores = GoldKnnDensityWeighingTool(k=1).weigh(x)

        assert scores.dtype == dtype
        assert scores.device == x.device
        assert torch.isfinite(scores).all()
        assert (scores > 0).all()

    def test_knn_density_respects_cosine_distance(self):
        x = torch.tensor([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]])

        scores = GoldKnnDensityWeighingTool(k=1, distance=DistanceType.COSINE).weigh(x)

        assert scores[-1] > scores[0]

    def test_same_sample_exclusion_selects_other_samples_and_matches_batches(self):
        x = torch.tensor([[0.0], [0.0], [0.0], [3.0], [10.0]])
        sample_idx = torch.tensor([0, 0, 1, 2, 2])
        expected = torch.tensor([1e-6, 1e-6, 1e-6, 3.0**0.5, 10.0**0.5])
        tool = GoldKnnDensityWeighingTool(k=1)
        one_batch = tool.weigh(x, sample_idx=sample_idx)
        small_batches = GoldKnnDensityWeighingTool(k=1, distance_batch_size=1).weigh(
            x, sample_idx=sample_idx
        )

        torch.testing.assert_close(one_batch, expected)
        torch.testing.assert_close(small_batches, one_batch)
        assert one_batch.dtype == x.dtype
        assert one_batch.device == x.device

        default = GoldKnnDensityWeighingTool(k=1)
        default_scores = default.weigh(x)
        torch.testing.assert_close(
            default.weigh(x, sample_idx=torch.arange(len(x))), default_scores
        )
        assert default_scores[0].item() == pytest.approx(1e-6)
        assert default_scores[4].item() == pytest.approx(7.0**0.5)
        assert one_batch[4].item() == pytest.approx(10.0**0.5)

        query_rows = torch.tensor([4, 4, 0])
        repeated = tool.weigh_batch(
            x,
            x[query_rows],
            sample_idx=sample_idx,
            batch_sample_idx=sample_idx[query_rows],
        )
        torch.testing.assert_close(repeated, one_batch[query_rows])
        empty = tool.weigh_batch(
            x,
            x[:0],
            sample_idx=sample_idx,
            batch_sample_idx=sample_idx[:0],
        )
        assert empty.shape == (0,)
        assert empty.dtype == x.dtype

        distinct = torch.tensor([[0.0], [4.0]])
        distinct_idx = torch.tensor([0, 1])
        with pytest.raises(ValueError, match=r"at least k \+ 2"):
            GoldKnnDensityWeighingTool(k=1).weigh(distinct)
        torch.testing.assert_close(
            GoldKnnDensityWeighingTool(k=1).weigh(distinct, sample_idx=distinct_idx),
            torch.tensor([2.0, 2.0]),
        )
        with pytest.raises(ValueError, match=r"at least k \+ 1"):
            GoldKnnDensityWeighingTool(k=2).weigh(distinct, sample_idx=distinct_idx)
        three = torch.tensor([[0.0], [4.0], [9.0]])
        torch.testing.assert_close(
            GoldKnnDensityWeighingTool(k=2).weigh(
                three, sample_idx=torch.tensor([0, 1, 2])
            ),
            torch.tensor([6.5, 4.5, 7.0]).sqrt(),
        )

    @pytest.mark.parametrize(
        "call,match",
        [
            pytest.param(
                lambda tool, x, ids: tool.weigh_batch(x, x, batch_sample_idx=ids),
                "sample_idx is required",
                id="missing-sample-idx",
            ),
            pytest.param(
                lambda tool, x, ids: tool.weigh_batch(x, x, sample_idx=ids),
                "batch_sample_idx is required",
                id="missing-query-sample-idx",
            ),
            pytest.param(
                lambda tool, x, ids: tool.weigh(x, sample_idx=ids[:-1]),
                "one entry per row",
                id="short-sample-idx",
            ),
            pytest.param(
                lambda tool, x, ids: tool.weigh(x, sample_idx=ids[:, None]),
                "1D integer tensor",
                id="sample-idx-rank",
            ),
            pytest.param(
                lambda tool, x, ids: tool.weigh(x, sample_idx=ids.to(torch.float32)),
                "1D integer tensor",
                id="float-sample-idx",
            ),
            pytest.param(
                lambda tool, x, ids: tool.weigh(x, sample_idx=ids.bool()),
                "1D integer tensor",
                id="bool-sample-idx",
            ),
            pytest.param(
                lambda tool, x, ids: tool.weigh(x, sample_idx=ids.tolist()),
                "1D integer tensor",
                id="list-sample-idx",
            ),
            pytest.param(
                lambda tool, x, ids: tool.weigh(
                    x[:3], sample_idx=torch.tensor([0, 0, 1])
                ),
                "eligible reference",
                id="too-few-eligible",
            ),
            pytest.param(
                lambda tool, x, ids: GoldKnnDensityWeighingTool(
                    k=2, distance_batch_size=1
                ).weigh(
                    torch.tensor([[0.0], [1.0], [2.0]]),
                    sample_idx=torch.tensor([1, 0, 0]),
                ),
                "query row 1 has 1",
                id="later-batch-shortfall",
            ),
        ],
    )
    def test_same_sample_exclusion_rejects_invalid_identities(self, call, match):
        x = torch.tensor([[0.0], [0.1], [5.0], [8.0]])
        sample_idx = torch.tensor([0, 0, 1, 1])
        tool = GoldKnnDensityWeighingTool(k=2)
        with pytest.raises(ValueError, match=match):
            call(tool, x, sample_idx)


class TestGoldDensityWeigherConfiguration:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"weight_key": "idx"}, "collides with the identity column"),
            ({"vectorized_key": "idx_vector"}, "collides with the identity column"),
            ({"weight_key": "vectorized"}, "share the same column name"),
            ({"label_key": "weight"}, "share the same column name"),
            (
                {"to_keep_schema": {"idx_vector": pxt.Int}},
                "cannot redefine the identity",
            ),
        ],
    )
    def test_rejects_column_name_collisions(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GoldDensityWeigher("unit_test.collisions", **kwargs)

    def test_distribution_and_constructor_limits_are_validated(self):
        with pytest.raises(NotImplementedError, match="Distributed processing"):
            GoldDensityWeigher("unit_test.distribute", distribute=True)
        weigher = GoldDensityWeigher("unit_test.distribute")
        with pytest.raises(NotImplementedError, match="Distributed processing"):
            weigher.distribute = True
        with pytest.raises(ValueError, match="positive integer"):
            GoldDensityWeigher("unit_test.invalid_chunk", chunk=0)
        with pytest.raises(ValueError, match="positive integer"):
            GoldDensityWeigher("unit_test.invalid_batch", weighing_batch_size=1.5)
        with pytest.raises(ValueError, match="exclude_same_sample must be a boolean"):
            GoldDensityWeigher("unit_test.invalid_exclude", exclude_same_sample=1)


@pytest.mark.usefixtures("pxt_unit_test_dir")
class TestGoldDensityWeigher:
    @pytest.mark.parametrize("source_type", ["dataset", "table"])
    @pytest.mark.parametrize(
        "include_vectorized_in_table", [False, True], ids=["omitted", "stored"]
    )
    def test_weigher_scores_each_prefix_and_preserves_existing_vector_scores(
        self, source_type, include_vectorized_in_table
    ):
        dataset = VectorDataset(
            [
                {
                    "idx": 20,
                    "idx_vector": 91,
                    "vectorized": torch.tensor([0.0, 0.0]),
                    "label": "a",
                },
                {
                    "idx": 20,
                    "idx_vector": 14,
                    "vectorized": torch.tensor([0.1, 0.0]),
                    "label": "a",
                },
                {
                    "idx": 50,
                    "idx_vector": 63,
                    "vectorized": torch.tensor([4.0, 0.0]),
                    "label": "b",
                },
                {
                    "idx": 50,
                    "idx_vector": 77,
                    "vectorized": torch.tensor([4.1, 0.0]),
                    "label": "b",
                },
            ]
        )
        source = dataset
        if source_type == "table":
            source = pxt.create_table(
                "unit_test.resumable_vectors",
                source=[
                    {**sample, "vectorized": sample["vectorized"].numpy()}
                    for sample in dataset.samples
                ],
                primary_key="idx_vector",
            )
        tool = CoordinateWeigher()
        weigher = GoldDensityWeigher(
            "unit_test.resumable_weights",
            weigher=tool,
            chunk=2,
            batch_size=2,
            max_batches=1,
            to_keep_schema={"label": pxt.String},
            include_vectorized_in_table=include_vectorized_in_table,
        )

        staged = weigher.setup_weighing_table(source)
        assert staged.count() == 2
        assert ("vectorized" in staged.columns()) == (
            source_type == "dataset" or include_vectorized_in_table
        )
        rows = staged.collect()
        assert {(row["idx"], row["idx_vector"], row["label"]) for row in rows} == {
            (20, 91, "a"),
            (20, 14, "a"),
        }
        if source_type == "dataset" or include_vectorized_in_table:
            expected_vectors = {
                sample["idx_vector"]: sample["vectorized"] for sample in dataset.samples
            }
            for row in rows:
                torch.testing.assert_close(
                    torch.as_tensor(row["vectorized"]),
                    expected_vectors[row["idx_vector"]],
                )
        assert all(row["weight"] is None for row in rows)
        assert tool.calls == 0
        assert weigher.setup_weighing_table(source).count() == 2

        partial = weigher.weigh_in_table(source)
        assert partial.count() == 2
        old_scores = weigher.get_weights_per_idx(
            partial, idx_key="idx_vector", normalize=False
        )
        assert len(weigher.weigh_in_dataset(source)) == 2

        resumed = GoldDensityWeigher(
            "unit_test.resumable_weights",
            weigher=CoordinateWeigher(),
            chunk=2,
            batch_size=2,
            max_batches=2,
            to_keep_schema={"label": pxt.String},
            include_vectorized_in_table=include_vectorized_in_table,
        )
        staged = resumed.setup_weighing_table(source)
        assert staged.count() == 4
        with pytest.raises(RuntimeError, match="Not all vectors have scores"):
            resumed.get_weights_per_idx(staged)
        assert resumed.weigher.calls == 0
        result = resumed.weigh_in_table(source)

        assert result.count() == 4
        sample_scores = resumed.get_weights_per_idx(result)
        vector_weights = resumed.get_weights_per_idx(
            result, idx_key="idx_vector", normalize=False
        )
        assert set(sample_scores) == {20, 50}
        assert len(vector_weights) == 4
        assert vector_weights == pytest.approx({91: 1.0, 14: 1.1, 63: 5.0, 77: 5.1})
        assert sample_scores == pytest.approx({20: 1.05 / 5.05, 50: 1.0})
        assert {idx: vector_weights[idx] for idx in old_scores} == old_scores
        assert {row["label"] for row in result.select(result.label).collect()} == {
            "a",
            "b",
        }

        with pytest.raises(ValueError, match="already exists"):
            GoldDensityWeigher(
                "unit_test.resumable_weights", allow_existing=False
            ).setup_weighing_table(source)

        new_tool = CoordinateWeigher()
        again = GoldDensityWeigher(
            "unit_test.resumable_weights",
            weigher=new_tool,
            chunk=2,
            batch_size=2,
            max_batches=2,
            to_keep_schema={"label": pxt.String},
            include_vectorized_in_table=include_vectorized_in_table,
        )
        again.weigh_in_table(source)
        assert new_tool.calls == 0

    def test_weigh_in_dataset_snapshot_survives_drop_table(
        self, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level("INFO", logger="goldener.weigh")
        dataset = VectorDataset(
            [
                {"idx": 0, "vectorized": torch.tensor([0.0, 1.0])},
                {"idx": 1, "vectorized": torch.tensor([1.0, 1.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.drop_weights", weigher=CountingWeigher(), drop_table=True
        )

        result = weigher.weigh_in_dataset(dataset)
        assert "Starting density weighing" in caplog.messages
        assert all("label 'None'" not in message for message in caplog.messages)

        assert len(result) == 2
        assert next(iter(result))["weight"] > 0
        assert isinstance(weigher.weigher, CountingWeigher)
        assert weigher.weigher.calls == 1
        assert weigher.weigher.batch_shapes == []
        with pytest.raises(pxt.Error, match="drop_weights"):
            pxt.get_table("unit_test.drop_weights")

    def test_empty_table_returns_empty_result(self):
        source = pxt.create_table(
            "unit_test.empty_vectors",
            schema={
                "idx": pxt.Int,
                "idx_vector": pxt.Required[pxt.Int],
                "vectorized": pxt.Array[(2,), pxt.Float],
            },
            primary_key="idx_vector",
        )
        weigher = GoldDensityWeigher("unit_test.empty_weights")

        result = weigher.weigh_in_table(source)

        assert result.count() == 0
        with pytest.raises(RuntimeError, match="No scores found"):
            weigher.get_weights_per_idx(result)

    def test_zero_max_batches_returns_an_empty_prefix_dataset(self):
        dataset = VectorDataset([{"idx": 0, "vectorized": torch.tensor([0.0])}])
        weigher = GoldDensityWeigher("unit_test.empty_prefix", max_batches=0)

        result = weigher.weigh_in_dataset(dataset)

        assert len(result) == 0

    @pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
    def test_sample_getter_averages_extreme_finite_scores(self, normalize):
        dataset = VectorDataset(
            [
                {"idx": 0, "vectorized": torch.tensor([0.0])},
                {"idx": 0, "vectorized": torch.tensor([1.0])},
                {"idx": 1, "vectorized": torch.tensor([2.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.extreme",
            weigher=FixedScoreWeigher([1e308, 1e308, 5e307]),
            chunk=None,
        )

        scores = weigher.get_weights_per_idx(
            weigher.weigh_in_table(dataset), normalize=normalize
        )

        assert scores == pytest.approx(
            {0: 1.0, 1: 0.5} if normalize else {0: 1e308, 1: 5e307}
        )

    @pytest.mark.parametrize("source_type", ["dataset", "table"])
    def test_getter_aggregates_by_sample_or_returns_vector_scores(self, source_type):
        dataset = VectorDataset(
            [
                {"idx": 0, "idx_vector": 21, "vectorized": torch.tensor([0.0])},
                {"idx": 0, "idx_vector": 22, "vectorized": torch.tensor([1.0])},
                {"idx": 1, "idx_vector": 23, "vectorized": torch.tensor([2.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.raw_sample_mean",
            weigher=FixedScoreWeigher([1.0, 3.0, 2.0]),
            chunk=None,
        )

        source = dataset
        if source_type == "table":
            source = pxt.create_table(
                "unit_test.raw_vectors",
                source=[
                    {**sample, "vectorized": sample["vectorized"].numpy()}
                    for sample in dataset.samples
                ],
                primary_key="idx_vector",
            )
        table = weigher.weigh_in_table(source)

        original = {row["idx_vector"]: row["weight"] for row in table.collect()}
        assert weigher.get_weights_per_idx(table, normalize=False) == {0: 2.0, 1: 2.0}
        assert weigher.get_weights_per_idx(table) == {0: 1.0, 1: 1.0}
        assert weigher.get_weights_per_idx(
            table, idx_key="idx_vector", normalize=False
        ) == {21: 1.0, 22: 3.0, 23: 2.0}
        assert weigher.get_weights_per_idx(
            table, idx_key="idx_vector"
        ) == pytest.approx({21: 1 / 3, 22: 1.0, 23: 2 / 3})
        assert weigher.get_weights_per_idx(table, aggregation=max) == pytest.approx(
            {0: 1.0, 1: 2 / 3}
        )
        assert weigher.get_weights_per_idx(table, max_raw_score=1.5) == {0: 1.0, 1: 1.0}
        assert weigher.get_weights_per_idx(
            table, normalize=False, max_raw_score=1.5
        ) == {0: 1.5, 1: 1.5}
        assert weigher.get_weights_per_idx(
            table, aggregation=max, normalize=False, max_raw_score=2.5
        ) == {0: 2.5, 1: 2.0}
        assert weigher.get_weights_per_idx(
            table, aggregation=max, normalize=False, max_raw_score=3.0
        ) == {0: 3.0, 1: 2.0}
        assert weigher.get_weights_per_idx(
            table, aggregation=max, max_raw_score=2.5
        ) == pytest.approx({0: 1.0, 1: 0.8})
        assert weigher.get_weights_per_idx(
            table, idx_key="idx_vector", max_raw_score=2.0
        ) == {21: 0.5, 22: 1.0, 23: 1.0}
        assert weigher.get_weights_per_idx(
            table, idx_key="idx_vector", normalize=False, max_raw_score=100.0
        ) == {21: 1.0, 22: 3.0, 23: 2.0}
        assert {row["idx_vector"]: row["weight"] for row in table.collect()} == original

    @pytest.mark.parametrize(
        "max_raw_score",
        [0.0, -1.0, float("nan"), float("inf"), True, "2.0"],
        ids=["zero", "negative", "nan", "inf", "bool", "string"],
    )
    def test_getter_rejects_invalid_max_raw_score(self, max_raw_score):
        weigher = GoldDensityWeigher(
            "unit_test.invalid_max_raw_score", weigher=FixedScoreWeigher([1.0, 2.0])
        )
        table = weigher.weigh_in_table(
            VectorDataset(
                [
                    {"idx": 0, "vectorized": torch.tensor([0.0])},
                    {"idx": 1, "vectorized": torch.tensor([1.0])},
                ]
            )
        )

        with pytest.raises(
            ValueError, match="max_raw_score must be finite and positive"
        ):
            weigher.get_weights_per_idx(table, max_raw_score=max_raw_score)

    def test_zero_raw_scores_remain_zero_when_normalized(self):
        dataset = VectorDataset(
            [
                {"idx": 0, "vectorized": torch.tensor([0.0])},
                {"idx": 1, "vectorized": torch.tensor([1.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.zero_scores", weigher=FixedScoreWeigher([0.0, 0.0])
        )

        sample_scores = weigher.get_weights_per_idx(
            weigher.weigh_in_table(dataset), normalize=False
        )

        assert sample_scores == {0: 0.0, 1: 0.0}
        assert (
            weigher.get_weights_per_idx(pxt.get_table(weigher.table_path))
            == sample_scores
        )

    @pytest.mark.parametrize("random_state", [None, 42], ids=["unseeded", "seeded"])
    def test_interrupted_chunk_resume_skips_committed_tool_calls(self, random_state):
        dataset = VectorDataset(
            [
                {"idx": index, "vectorized": torch.tensor([float(index)])}
                for index in range(4)
            ]
        )
        table_path = "unit_test.interrupted"
        first_tool = InterruptingWeigher(interrupt=True)
        first = GoldDensityWeigher(
            table_path,
            weigher=first_tool,
            chunk=2,
            batch_size=2,
            random_state=random_state,
        )

        with pytest.raises(RuntimeError, match="simulated interruption"):
            first.weigh_in_table(dataset)
        assert first_tool.calls == 2
        interrupted_table = pxt.get_table(table_path)
        computed = {
            row["idx_vector"]: row["weight"]
            for row in interrupted_table.collect()
            if row["weight"] is not None
        }
        assert len(computed) == 2
        with pytest.raises(RuntimeError, match="Not all vectors have scores"):
            GoldDensityWeigher.get_weights_per_idx(interrupted_table)

        resumed_tool = InterruptingWeigher(interrupt=False)
        resumed = GoldDensityWeigher(
            table_path,
            weigher=resumed_tool,
            chunk=2,
            batch_size=2,
            random_state=random_state,
        )
        result = resumed.weigh_in_table(dataset)

        assert resumed_tool.calls == 1
        scores = resumed.get_weights_per_idx(
            result, idx_key="idx_vector", normalize=False
        )
        assert scores == {idx: float(idx + 1) for idx in range(4)}
        assert {idx: scores[idx] for idx in computed} == computed

    @pytest.mark.parametrize(
        "chunk", [None, 1, 100], ids=["full", "single", "oversized"]
    )
    def test_table_and_dataset_inputs_preserve_custom_keys_and_sequence_metadata(
        self, chunk
    ):
        samples = [
            {
                "idx": 4,
                "idx_vector": 71,
                "features": torch.tensor([0.0, 1.0]),
                "label_seq": torch.tensor([2, 3]),
            },
            {
                "idx": 9,
                "idx_vector": 33,
                "features": torch.tensor([3.0, 1.0]),
                "label_seq": torch.tensor([5, 8]),
            },
        ]
        dataset = VectorDataset(samples)
        source = pxt.create_table(
            "unit_test.custom_vectors",
            source=[
                {
                    "idx": sample["idx"],
                    "idx_vector": sample["idx_vector"],
                    "features": sample["features"].numpy(),
                    "label_seq": sample["label_seq"].numpy(),
                }
                for sample in samples
            ],
            primary_key="idx_vector",
        )
        options = {
            "vectorized_key": "features",
            "weight_key": "density",
            "to_keep_schema": {"label_seq": pxt.Array[(2,), pxt.Int]},
            "collate_fn": custom_collate,
            "weigher": CoordinateWeigher(),
            "chunk": chunk,
        }
        table_weigher = GoldDensityWeigher("unit_test.table_weights", **options)
        dataset_weigher = GoldDensityWeigher("unit_test.dataset_weights", **options)

        from_table = table_weigher.weigh_in_table(source)
        from_dataset = dataset_weigher.weigh_in_table(dataset)

        assert table_weigher.get_weights_per_idx(
            from_table, "density"
        ) == pytest.approx(dataset_weigher.get_weights_per_idx(from_dataset, "density"))
        dataset_view = dataset_weigher.weigh_in_dataset(dataset)
        items = list(dataset_view)
        assert {
            int(item["idx_vector"]): (
                int(item["idx"]),
                float(item["density"]),
                tuple(item["label_seq"].tolist()),
            )
            for item in items
        } == {71: (4, 1.0, (2, 3)), 33: (9, 4.0, (5, 8))}
        assert {
            row["idx_vector"]: (row["idx"], row["density"], tuple(row["label_seq"]))
            for row in from_table.collect()
        } == {71: (4, 1.0, (2, 3)), 33: (9, 4.0, (5, 8))}
        assert "density_chunk" not in items[0]

    @pytest.mark.parametrize(
        "values",
        [[1.0, 2.0, 3.0], [float("nan"), 1.0], [-1.0, 1.0]],
        ids=["wrong-score-count", "nonfinite-score", "negative-score"],
    )
    @pytest.mark.parametrize("weighing_batch_size", [None, 1])
    def test_weigher_rejects_malformed_custom_tool_outputs(
        self, values, weighing_batch_size
    ):
        dataset = VectorDataset(
            [
                {"idx": 0, "vectorized": torch.tensor([0.0])},
                {"idx": 1, "vectorized": torch.tensor([1.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.bad_tool",
            weigher=FixedScoreWeigher(values),
            weighing_batch_size=weighing_batch_size,
            chunk=None,
            batch_size=2,
        )

        with pytest.raises(ValueError, match="one finite nonnegative score"):
            weigher.weigh_in_table(dataset)

    @pytest.mark.parametrize("invalid_schema", ["primary-key", "missing-idx"])
    def test_existing_result_table_must_have_required_schema(self, invalid_schema):
        pxt.create_table(
            "unit_test.wrong_pk",
            source=[
                {
                    **({"idx": 0} if invalid_schema == "primary-key" else {}),
                    "idx_vector": 0,
                    "vectorized": [0.0],
                }
            ],
            primary_key="idx" if invalid_schema == "primary-key" else "idx_vector",
        )
        match = (
            "idx_vector"
            if invalid_schema == "primary-key"
            else "missing required keys.*idx"
        )
        with pytest.raises(ValueError, match=match):
            GoldDensityWeigher("unit_test.wrong_pk").weigh_in_table(VectorDataset([]))

    @pytest.mark.parametrize(
        "label_key", [None, "label"], ids=["whole-population", "per-label"]
    )
    @pytest.mark.parametrize("weighing_batch_size", [None, 1, 4])
    def test_weigher_uses_independent_label_populations_and_reuses_scores(
        self, label_key, weighing_batch_size
    ):
        vectors = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.1, 0.0], [10.0, 0.0], [30.0, 0.0]]
        )
        labels = ["a"] * 3 + ["b"] * 3
        source = VectorDataset(
            [
                {"idx": idx, "vectorized": vector, "label": label}
                for idx, (vector, label) in enumerate(zip(vectors, labels))
            ]
        )
        tool = GoldKnnDensityWeighingTool(k=1)
        weigher = GoldDensityWeigher(
            "unit_test.label_weights",
            weigher=tool,
            label_key=label_key,
            batch_size=3,
            weighing_batch_size=weighing_batch_size,
            to_keep_schema={"label": pxt.String},
        )

        result = weigher.weigh_in_table(source)
        expected = (
            torch.cat([tool.weigh(vectors[:3]), tool.weigh(vectors[3:])])
            if label_key is not None
            else tool.weigh(vectors)
        )
        scores = weigher.get_weights_per_idx(
            result, idx_key="idx_vector", normalize=False
        )
        assert scores == pytest.approx(dict(enumerate(expected.tolist())))
        weigher.weigher = InterruptingWeigher(interrupt=True)
        weigher.weigh_in_table(source)
        assert weigher.weigher.calls == 0
        assert (
            weigher.get_weights_per_idx(result, idx_key="idx_vector", normalize=False)
            == scores
        )

        selected = {idx: score for idx, score in scores.items() if idx < 3}
        assert weigher.get_weights_per_idx(
            result,
            idx_key="idx_vector",
            label_key="label",
            label_value="a",
            normalize=False,
        ) == pytest.approx(selected)
        assert weigher.get_weights_per_idx(
            result, label_key="label", label_value="a"
        ) == pytest.approx(
            {idx: score / max(selected.values()) for idx, score in selected.items()}
        )
        selected_b = {idx: score for idx, score in scores.items() if idx >= 3}
        assert weigher.get_weights_per_idx(result, label_key="label") == pytest.approx(
            {
                **{
                    idx: score / max(selected.values())
                    for idx, score in selected.items()
                },
                **{
                    idx: score / max(selected_b.values())
                    for idx, score in selected_b.items()
                },
            }
        )
        with pytest.raises(RuntimeError, match="No scores found"):
            weigher.get_weights_per_idx(result, label_key="label", label_value="absent")

    def test_getter_normalizes_each_label_population_when_label_value_is_omitted(self):
        table = pxt.create_table(
            "unit_test.per_label_weights",
            source=[
                {"idx": 10, "idx_vector": 0, "weight": 10.0, "label": "a"},
                {"idx": 10, "idx_vector": 1, "weight": 30.0, "label": "a"},
                {"idx": 11, "idx_vector": 2, "weight": 40.0, "label": "a"},
                {"idx": 20, "idx_vector": 3, "weight": 0.0, "label": "b"},
                {"idx": 20, "idx_vector": 4, "weight": 0.0, "label": "b"},
                {"idx": 21, "idx_vector": 5, "weight": 2.0, "label": "b"},
            ],
            primary_key="idx_vector",
        )

        per_label = GoldDensityWeigher.get_weights_per_idx(table, label_key="label")
        assert per_label == pytest.approx({10: 0.5, 11: 1.0, 20: 0.0, 21: 1.0})

        merged = {
            **GoldDensityWeigher.get_weights_per_idx(
                table, label_key="label", label_value="a"
            ),
            **GoldDensityWeigher.get_weights_per_idx(
                table, label_key="label", label_value="b"
            ),
        }
        assert per_label == pytest.approx(merged)

    @pytest.mark.parametrize("reducer_type", [TrackingReducer, FittedTrackingReducer])
    @pytest.mark.parametrize("weighing_batch_size", [None, 1, 2])
    def test_reduction_runs_per_chunk_before_batched_scoring(
        self, reducer_type, weighing_batch_size
    ):
        source = VectorDataset(
            [{"vectorized": torch.tensor([float(i), 100.0 + i])} for i in range(6)]
        )
        reducer = reducer_type()
        tool = CoordinateWeigher()
        tool.weigh_batch = Mock(wraps=tool.weigh_batch)
        weigher = GoldDensityWeigher(
            "unit_test.reduced",
            weigher=tool,
            reducer=reducer,
            chunk=3,
            batch_size=2,
            weighing_batch_size=weighing_batch_size,
            random_state=7,
        )
        result = weigher.weigh_in_table(source)
        assert {row["idx_vector"]: row["idx"] for row in result.collect()} == dict(
            enumerate(range(6))
        )
        assert weigher.get_weights_per_idx(result, normalize=False) == {
            i: float(2 * i + 1) for i in range(6)
        }
        assert len(reducer.inputs) == 2
        assert all(x.shape == (3, 2) for x in reducer.inputs)
        if isinstance(reducer, FittedTrackingReducer):
            assert len(reducer.fitted_inputs) == 2
            for fitted, transformed in zip(reducer.fitted_inputs, reducer.inputs):
                torch.testing.assert_close(fitted, transformed)
        if weighing_batch_size is None:
            tool.weigh_batch.assert_not_called()
        else:
            calls = iter(tool.weigh_batch.call_args_list)
            for chunk in reducer.inputs:
                reduced = chunk[:, :1] * 2
                for batch in reduced.split(weighing_batch_size):
                    reference, query = next(calls).args
                    torch.testing.assert_close(reference, reduced)
                    torch.testing.assert_close(query, batch)
            assert next(calls, None) is None
        previous_calls = tool.calls
        weigher.weigh_in_table(source)
        assert tool.calls == previous_calls
        assert len(reducer.inputs) == 2

    @pytest.mark.parametrize(
        "output",
        [None, torch.ones(2), torch.ones(1, 1)],
        ids=["not-a-tensor", "not-2d", "changed-row-count"],
    )
    def test_rejects_malformed_reducer_output_before_scoring(self, output):
        reducer = Mock(spec=GoldReductionTool)
        reducer.transform.return_value = output
        tool = CountingWeigher()
        weigher = GoldDensityWeigher(
            "unit_test.bad_reducer", weigher=tool, reducer=reducer
        )
        source = VectorDataset(
            [{"vectorized": torch.tensor([float(i)])} for i in range(2)]
        )
        with pytest.raises(
            ValueError,
            match="Reducer output must be 2D and preserve the number of rows",
        ):
            weigher.weigh_in_table(source)
        assert tool.calls == 0
        assert all(
            row["weight"] is None for row in pxt.get_table(weigher.table_path).collect()
        )

    @pytest.mark.parametrize("source_type", ["dataset", "table", "non-dict"])
    def test_rejects_invalid_source(self, source_type):
        source = VectorDataset([{"other": torch.tensor([0.0])}])
        match = "Sample is missing expected keys.*vectorized"
        if source_type == "table":
            source = pxt.create_table(
                "unit_test.missing_vectors", source=[{"idx": 0, "idx_vector": 0}]
            )
            match = "Source table does not contain vector column 'vectorized'"
        elif source_type == "non-dict":
            source = VectorDataset([torch.tensor([0.0])])
            match = "Sample must be a dictionary"
        with pytest.raises(ValueError, match=match):
            GoldDensityWeigher("unit_test.invalid_source").weigh_in_table(source)

    @pytest.mark.parametrize(
        "chunk,label_key",
        [(2, None), (None, "label")],
        ids=["small-chunks", "small-label-group"],
    )
    def test_knn_rejects_insufficient_chunk_or_label_populations(
        self, chunk, label_key
    ):
        source = VectorDataset(
            [
                {
                    "idx": i,
                    "vectorized": torch.tensor([float(i)]),
                    "label": "a" if i < 2 else "b",
                }
                for i in range(4)
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.small_population",
            weigher=GoldKnnDensityWeighingTool(k=1),
            chunk=chunk,
            label_key=label_key,
        )
        with pytest.raises(ValueError, match=r"at least k \+ 2"):
            weigher.weigh_in_table(source)
        assert all(
            row["weight"] is None for row in pxt.get_table(weigher.table_path).collect()
        )

    def test_same_sample_exclusion_scores_below_the_default_population_minimum(self):
        source = VectorDataset(
            [
                {"idx": 0, "idx_vector": 4, "vectorized": torch.tensor([0.0])},
                {"idx": 1, "idx_vector": 9, "vectorized": torch.tensor([4.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.same_sample_minimum",
            weigher=GoldKnnDensityWeighingTool(k=1),
            exclude_same_sample=True,
        )
        result = weigher.weigh_in_table(source)
        assert weigher.get_weights_per_idx(
            result, idx_key="idx_vector", normalize=False
        ) == pytest.approx({4: 2.0, 9: 2.0})

    @pytest.mark.parametrize("source_type", ["dataset", "table"])
    def test_same_sample_exclusion_rejects_short_orchestration_populations(
        self, source_type
    ):
        samples = [
            {"idx": 0, "idx_vector": 1, "vectorized": torch.tensor([0.0])},
            {"idx": 0, "idx_vector": 2, "vectorized": torch.tensor([0.1])},
            {"idx": 1, "idx_vector": 3, "vectorized": torch.tensor([4.0])},
        ]
        source = _vector_source(samples, source_type, "unit_test.same_sample_short_src")
        weigher = GoldDensityWeigher(
            "unit_test.same_sample_short",
            weigher=GoldKnnDensityWeighingTool(k=2),
            exclude_same_sample=True,
        )
        with pytest.raises(ValueError, match="eligible reference"):
            weigher.weigh_in_table(source)
        assert all(
            row["weight"] is None for row in pxt.get_table(weigher.table_path).collect()
        )

    @pytest.mark.parametrize("source_type", ["dataset", "table"])
    @pytest.mark.parametrize("weighing_batch_size", [None, 1])
    @pytest.mark.parametrize("use_reducer", [False, True])
    def test_same_sample_exclusion_keeps_idx_aligned_through_orchestration(
        self, source_type, weighing_batch_size, use_reducer
    ):
        samples = [
            {
                "idx": sample,
                "idx_vector": idx_vector,
                "vectorized": torch.tensor([value]),
            }
            for idx_vector, sample, value in (
                (10, 0, 0.0),
                (11, 0, 0.1),
                (20, 1, 10.0),
                (21, 1, 10.1),
                (30, 2, 20.0),
                (31, 2, 20.1),
            )
        ]
        source = _vector_source(samples, source_type, "unit_test.same_sample_vectors")
        weigher = GoldDensityWeigher(
            "unit_test.same_sample_weights",
            weigher=GoldKnnDensityWeighingTool(k=1),
            exclude_same_sample=True,
            chunk=3,
            random_state=1,
            weighing_batch_size=weighing_batch_size,
            reducer=TrackingReducer() if use_reducer else None,
            include_vectorized_in_table=False,
            batch_size=2,
        )
        result = weigher.weigh_in_table(source)
        assert len(weigher.weigh_in_dataset(source)) == len(samples)
        stored = weigher.get_weights_per_idx(
            result, idx_key="idx_vector", normalize=False
        )
        assert stored == pytest.approx(
            _expected_same_sample_scores(samples, reduced=use_reducer)
        )
        repeated = weigher.get_weights_per_idx(
            weigher.weigh_in_table(source), idx_key="idx_vector", normalize=False
        )
        assert repeated == pytest.approx(stored)


def _vector_source(samples, source_type, table_path):
    if source_type == "dataset":
        return VectorDataset(samples)
    return pxt.create_table(
        table_path,
        source=[
            {**sample, "vectorized": sample["vectorized"].numpy()} for sample in samples
        ],
        primary_key="idx_vector",
    )


def _expected_same_sample_scores(samples, reduced):
    by_vector = {
        sample["idx_vector"]: (sample["vectorized"], sample["idx"])
        for sample in samples
    }
    chunks = get_random_chunk_assignment(
        sorted(by_vector), max_chunk_size=3, random_state=1
    )
    tool = GoldKnnDensityWeighingTool(k=1)
    expected = {}
    for group in chunks:
        rows = [by_vector[idx] for idx in group]
        vectors = torch.stack([row[0] for row in rows])
        if reduced:
            vectors = vectors[:, :1] * 2
        scores = tool.weigh(vectors, sample_idx=torch.tensor([row[1] for row in rows]))
        for idx_vector, score in zip(group, scores):
            expected[idx_vector] = float(score)
    return expected
