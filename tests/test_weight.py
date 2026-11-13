import pytest
import pixeltable as pxt
import torch
from torch.utils.data import Dataset

from goldener.select import DistanceType
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
        return self.samples[index].copy()


def custom_collate(batch):
    return collate_keeping_sequences_as_sequences(batch)


class CountingWeigher(GoldDensityWeighingTool):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.batch_shapes = []

    def weigh_batch(self, x, batch):
        self.batch_shapes.append((len(x), len(batch)))
        return self.weigh(batch)

    def weigh(self, x):
        self.calls += 1
        return x.new_ones((len(x),))


class FixedScoreWeigher(GoldDensityWeighingTool):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def weigh_batch(self, x, batch):
        scores = self.weigh(x)
        if scores.shape != (len(x),):
            return scores
        indices = (batch[:, None, :] == x[None, :, :]).all(dim=2).long().argmax(dim=1)
        return scores[indices]

    def weigh(self, x):
        if isinstance(self.values, Exception):
            raise self.values
        return torch.tensor(self.values, dtype=torch.float64)


class InterruptingWeigher(CountingWeigher):
    def __init__(self, interrupt):
        super().__init__()
        self.interrupt = interrupt
        self.calls = 0

    def weigh(self, x):
        self.calls += 1
        if self.interrupt and self.calls == 2:
            raise RuntimeError("simulated interruption")
        return x.new_ones((len(x),))


@pytest.fixture
def pxt_unit_test_dir():
    pxt.drop_dir("unit_test", force=True)
    pxt.create_dir("unit_test", if_exists="ignore")
    yield
    pxt.drop_dir("unit_test", force=True)


class TestGoldKnnDensityWeighingTool:
    def test_knn_density_returns_raw_distance_power_and_sparse_points_score_higher(
        self,
    ):
        x = torch.tensor([[0.0], [0.1], [0.2], [5.0]])
        scores = GoldKnnDensityWeighingTool(k=1).weigh(x)

        assert scores.shape == (4,)
        assert scores.dtype == x.dtype
        assert scores.device == x.device
        assert scores[0].item() == pytest.approx(0.1**0.5)
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

    @pytest.mark.parametrize("row_indices", [[3, 0, 2, 1], [2], [1, 1, 3], []])
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
        "kwargs",
        [
            {"k": 0},
            {"k": True},
            {"distance_batch_size": 0},
            {"power": -1},
            {"power": float("nan")},
            {"power": float("inf")},
            {"power": True},
            {"eps": 0},
            {"eps": float("nan")},
            {"eps": float("inf")},
            {"eps": True},
        ],
    )
    def test_knn_density_rejects_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            GoldKnnDensityWeighingTool(**kwargs)

    @pytest.mark.parametrize(
        "x",
        [
            torch.ones(3),
            torch.empty((2, 0)),
            torch.tensor([[float("nan")]]),
            torch.ones((7, 2), dtype=torch.int64),
        ],
    )
    def test_knn_density_rejects_invalid_input(self, x):
        with pytest.raises(ValueError):
            GoldKnnDensityWeighingTool().weigh(x)
        with pytest.raises(ValueError):
            GoldKnnDensityWeighingTool().weigh_batch(torch.zeros(7, 2), x)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
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


@pytest.mark.usefixtures("pxt_unit_test_dir")
class TestGoldDensityWeigher:
    @pytest.mark.parametrize(
        "include_vectorized", [False, True], ids=["omitted", "stored"]
    )
    def test_add_weighing_rows_optionally_stores_vectors(self, include_vectorized):
        dataset = VectorDataset(
            [
                {
                    "idx": 4,
                    "idx_vector": 71,
                    "features": torch.tensor([0.0, 1.0]),
                    "label": "a",
                }
            ]
        )
        schema = {
            "idx": pxt.Int,
            "idx_vector": pxt.Required[pxt.Int],
            "label": pxt.String,
            "weight": pxt.Float,
        }
        if include_vectorized:
            schema["features"] = pxt.Array[(2,), pxt.Float]
        table = pxt.create_table(
            "unit_test.optional_vectors", schema=schema, primary_key="idx_vector"
        )
        weigher = GoldDensityWeigher(
            "unit_test.optional_vectors",
            vectorized_key="features",
            to_keep_schema={"label": pxt.String},
        )
        options = {"include_vectorized": True} if include_vectorized else {}

        weigher._add_rows_to_weighing_table_from_dataset(dataset, table, **options)

        rows = table.collect()
        assert len(rows) == 1
        row = rows[0]
        assert (row["idx"], row["idx_vector"], row["label"]) == (4, 71, "a")
        assert ("features" in row) == include_vectorized
        if include_vectorized:
            assert torch.equal(torch.as_tensor(row["features"]), dataset[0]["features"])

    @pytest.mark.parametrize("source_type", ["dataset", "table"])
    @pytest.mark.parametrize("include_vectorized_in_table", [False, True])
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
        tool = CountingWeigher()
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
            weigher=CountingWeigher(),
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
        assert vector_weights[91] == vector_weights[14] == sample_scores[20]
        assert vector_weights[63] == vector_weights[77] == sample_scores[50]
        assert {idx: vector_weights[idx] for idx in old_scores} == old_scores
        assert {row["label"] for row in result.select(result.label).collect()} == {
            "a",
            "b",
        }

        with pytest.raises(ValueError, match="already exists"):
            GoldDensityWeigher(
                "unit_test.resumable_weights", allow_existing=False
            ).setup_weighing_table(source)

        # A new doer reuses completed weights without calling its tensor tool.
        new_tool = CountingWeigher()
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

    def test_weigh_in_dataset_snapshot_survives_drop_table(self):
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

        assert len(result) == 2
        assert next(iter(result))["weight"] > 0
        assert weigher.weigher.calls == 1
        assert weigher.weigher.batch_shapes == []
        with pytest.raises(pxt.Error):
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

    def test_sample_getter_averages_extreme_finite_raw_scores_without_normalizing(self):
        dataset = VectorDataset(
            [
                {"idx": 0, "vectorized": torch.tensor([0.0])},
                {"idx": 1, "vectorized": torch.tensor([1.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.extreme",
            weigher=FixedScoreWeigher([1e308, 5e307]),
            chunk=None,
        )

        scores = weigher.get_weights_per_idx(
            weigher.weigh_in_table(dataset), normalize=False
        )

        assert scores[0] == 1e308
        assert scores[1] == 5e307

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
        assert {row["idx_vector"]: row["weight"] for row in table.collect()} == original

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
        assert len(scores) == 4
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
            "weigher": CountingWeigher(),
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
        assert {tuple(item["label_seq"].tolist()) for item in items} == {(2, 3), (5, 8)}
        assert "density_chunk" not in items[0]

    @pytest.mark.parametrize("values", [[1.0], [float("nan"), 1.0], [-1.0, 1.0]])
    def test_weigher_rejects_malformed_custom_tool_outputs(self, values):
        dataset = VectorDataset(
            [
                {"idx": 0, "vectorized": torch.tensor([0.0])},
                {"idx": 1, "vectorized": torch.tensor([1.0])},
            ]
        )
        weigher = GoldDensityWeigher(
            "unit_test.bad_tool",
            weigher=FixedScoreWeigher(values),
            chunk=None,
            batch_size=2,
        )

        with pytest.raises(ValueError, match="one finite nonnegative score"):
            weigher.weigh_in_table(dataset)

    def test_distribution_and_constructor_limits_are_validated(self):
        with pytest.raises(NotImplementedError, match="Distributed processing"):
            GoldDensityWeigher("unit_test.distribute", distribute=True)
        weigher = GoldDensityWeigher("unit_test.distribute")
        with pytest.raises(NotImplementedError, match="Distributed processing"):
            weigher.distribute = True
        with pytest.raises(ValueError, match="positive integer"):
            GoldDensityWeigher("unit_test.invalid_chunk", chunk=0)

    def test_existing_result_table_must_have_idx_vector_primary_key(self):
        pxt.create_table(
            "unit_test.wrong_pk",
            source=[
                {
                    "idx": 0,
                    "idx_vector": 0,
                    "vectorized": [0.0],
                }
            ],
            primary_key="idx",
        )
        with pytest.raises(ValueError, match="idx_vector"):
            GoldDensityWeigher("unit_test.wrong_pk").weigh_in_table(VectorDataset([]))

    @pytest.mark.parametrize("label_key", [None, "label"])
    def test_weigher_uses_independent_label_populations_and_reuses_scores(
        self, label_key
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
            "unit_test.label_weights", weigher=tool, label_key=label_key, batch_size=3
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
