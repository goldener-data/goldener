import pytest
import pixeltable as pxt
import torch
from torch.utils.data import Dataset

from goldener.select import DistanceType
from goldener.torch_utils import collate_keeping_sequences_as_sequences
from goldener.weight import (
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

    def weigh(self, x):
        self.calls += 1
        return x.new_ones((len(x),))


class FixedScoreWeigher(GoldDensityWeighingTool):
    def __init__(self, values):
        super().__init__()
        self.values = values

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


def test_knn_density_returns_raw_distance_power_and_sparse_points_score_higher():
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
def test_knn_density_rejects_insufficient_populations(k, power, size_offset):
    size = k + 1 if size_offset == 2 else size_offset
    with pytest.raises(ValueError, match=r"at least k \+ 2"):
        GoldKnnDensityWeighingTool(k=k, power=power).weigh(torch.zeros(size, 2))


@pytest.mark.parametrize("k", [1, 5])
def test_knn_density_zero_power_at_minimum_population(k):
    scores = GoldKnnDensityWeighingTool(k=k, power=0).weigh(torch.zeros(k + 2, 2))
    assert torch.equal(scores, torch.ones(k + 2))


def test_knn_density_uses_absolute_eps_for_identical_vectors():
    scores = GoldKnnDensityWeighingTool(power=0.5, eps=1e-8).weigh(torch.zeros((7, 2)))

    assert torch.equal(scores, torch.full((7,), 1e-4))


def test_knn_density_excludes_only_self_and_query_batching_preserves_results():
    x = torch.tensor([[0.0], [0.0], [3.0], [8.0]])
    one_batch = GoldKnnDensityWeighingTool(k=1, distance_batch_size=20).weigh(x)
    small_batches = GoldKnnDensityWeighingTool(k=1, distance_batch_size=1).weigh(x)

    assert torch.allclose(one_batch, small_batches)
    assert one_batch[0] == one_batch[1]
    assert one_batch[0].item() == pytest.approx(1e-6)
    assert one_batch[2] > one_batch[0]


def test_knn_density_preserves_small_distances_in_large_dynamic_range():
    x = torch.tensor([[0.0], [1.0], [1e30]], dtype=torch.float32)

    scores = GoldKnnDensityWeighingTool(k=1).weigh(x)

    assert torch.allclose(scores, torch.tensor([1.0, 1.0, 1e15]))


def test_knn_raw_scores_scale_in_original_distance_units():
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
def test_knn_density_rejects_invalid_configuration(kwargs):
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
def test_knn_density_rejects_invalid_input(x):
    with pytest.raises(ValueError):
        GoldKnnDensityWeighingTool().weigh(x)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_knn_density_preserves_supported_float_dtypes(dtype):
    x = torch.tensor([[0.0, 1.0], [0.1, 1.0], [4.0, 1.0]], dtype=dtype)

    scores = GoldKnnDensityWeighingTool(k=1).weigh(x)

    assert scores.dtype == dtype
    assert scores.device == x.device
    assert torch.isfinite(scores).all()
    assert (scores > 0).all()


def test_knn_density_respects_cosine_distance():
    x = torch.tensor([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]])

    scores = GoldKnnDensityWeighingTool(k=1, distance=DistanceType.COSINE).weigh(x)

    assert scores[-1] > scores[0]


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_weigher_scores_each_prefix_and_preserves_existing_vector_scores():
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
    tool = CountingWeigher()
    weigher = GoldDensityWeigher(
        "unit_test.resumable_weights",
        weigher=tool,
        chunk=2,
        batch_size=2,
        max_batches=1,
        to_keep_schema={"label": pxt.String},
    )

    partial = weigher.weigh_in_table(dataset)
    assert partial.count() == 2
    old_scores = weigher.get_weights_per_idx_vector(partial)
    old_chunks = {
        row["idx_vector"]: row["density_chunk"]
        for row in partial.select(partial.idx_vector, partial.density_chunk).collect()
    }
    assert len(weigher.weigh_in_dataset(dataset)) == 2

    resumed = GoldDensityWeigher(
        "unit_test.resumable_weights",
        weigher=CountingWeigher(),
        chunk=2,
        batch_size=2,
        max_batches=2,
        to_keep_schema={"label": pxt.String},
    )
    result = resumed.weigh_in_table(dataset)

    assert result.count() == 4
    sample_scores = resumed.get_weights_per_idx(result)
    vector_weights = resumed.get_weights_per_idx_vector(result)
    assert set(sample_scores) == {20, 50}
    assert len(vector_weights) == 4
    assert vector_weights[91] == vector_weights[14] == sample_scores[20]
    assert vector_weights[63] == vector_weights[77] == sample_scores[50]
    assert {idx: vector_weights[idx] for idx in old_scores} == old_scores
    new_chunks = {
        row["idx_vector"]: row["density_chunk"]
        for row in result.select(result.idx_vector, result.density_chunk).collect()
    }
    assert {idx: new_chunks[idx] for idx in old_chunks} == old_chunks
    assert {new_chunks[idx] for idx in (63, 77)}.isdisjoint(old_chunks.values())
    assert {row["label"] for row in result.select(result.label).collect()} == {"a", "b"}

    with pytest.raises(ValueError, match="already exists"):
        GoldDensityWeigher(
            "unit_test.resumable_weights", allow_existing=False
        ).weigh_in_table(dataset)

    # A new doer reuses completed weights without calling its tensor tool.
    new_tool = CountingWeigher()
    again = GoldDensityWeigher(
        "unit_test.resumable_weights",
        weigher=new_tool,
        chunk=2,
        batch_size=2,
        max_batches=2,
        to_keep_schema={"label": pxt.String},
    )
    again.weigh_in_table(dataset)
    assert new_tool.calls == 0


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_weigh_in_dataset_snapshot_survives_drop_table():
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
    with pytest.raises(pxt.Error):
        pxt.get_table("unit_test.drop_weights")


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_empty_table_returns_empty_result():
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
    assert weigher.get_weights_per_idx(result) == {}


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_zero_max_batches_returns_an_empty_prefix_dataset():
    dataset = VectorDataset([{"idx": 0, "vectorized": torch.tensor([0.0])}])
    weigher = GoldDensityWeigher("unit_test.empty_prefix", max_batches=0)

    result = weigher.weigh_in_dataset(dataset)

    assert len(result) == 0


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_normalization_is_explicit_nonmutating_and_uses_unique_sample_scores():
    dataset = VectorDataset(
        [
            {"idx": 10, "vectorized": torch.tensor([0.0])},
            {"idx": 10, "vectorized": torch.tensor([0.1])},
            {"idx": 20, "vectorized": torch.tensor([10.0])},
        ]
    )
    weigher = GoldDensityWeigher(
        "unit_test.unequal", weigher=GoldKnnDensityWeighingTool(k=1), chunk=None
    )

    result = weigher.weigh_in_table(dataset)
    scores = weigher.get_weights_per_idx(result)
    original_vector_scores = weigher.get_weights_per_idx_vector(result)
    weights = weigher.normalize_weights(scores)

    assert scores[20] > scores[10]
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)
    assert weigher.get_weights_per_idx_vector(result) == original_vector_scores


@pytest.mark.parametrize(
    "scores,expected",
    [({}, {}), ({1: 0.0, 2: 0.0}, {1: 1.0, 2: 1.0})],
)
def test_normalize_weights_empty_and_zero_inputs(scores, expected):
    original = scores.copy()

    assert GoldDensityWeigher.normalize_weights(scores) == expected
    assert scores == original


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_normalize_weights_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GoldDensityWeigher.normalize_weights({1: value})


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_sample_getter_averages_extreme_finite_raw_scores_without_normalizing():
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

    scores = weigher.get_weights_per_idx(weigher.weigh_in_table(dataset))

    assert scores[0] == 1e308
    assert scores[1] == 5e307


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_sample_getter_averages_raw_vectors_while_vector_getter_keeps_them():
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

    table = weigher.weigh_in_table(dataset)

    assert weigher.get_weights_per_idx(table) == {0: 2.0, 1: 2.0}
    assert weigher.get_weights_per_idx_vector(table) == {21: 1.0, 22: 3.0, 23: 2.0}


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_zero_raw_scores_are_valid_and_normalize_to_uniform_weights():
    dataset = VectorDataset(
        [
            {"idx": 0, "vectorized": torch.tensor([0.0])},
            {"idx": 1, "vectorized": torch.tensor([1.0])},
        ]
    )
    weigher = GoldDensityWeigher(
        "unit_test.zero_scores", weigher=FixedScoreWeigher([0.0, 0.0])
    )

    sample_scores = weigher.get_weights_per_idx(weigher.weigh_in_table(dataset))

    assert sample_scores == {0: 0.0, 1: 0.0}
    assert weigher.normalize_weights(sample_scores) == {0: 1.0, 1: 1.0}


@pytest.mark.usefixtures("pxt_unit_test_dir")
@pytest.mark.parametrize("random_state", [None, 42], ids=["unseeded", "seeded"])
def test_interrupted_chunk_resume_skips_committed_tool_calls(random_state):
    dataset = VectorDataset(
        [
            {"idx": index, "vectorized": torch.tensor([float(index)])}
            for index in range(4)
        ]
    )
    table_path = "unit_test.interrupted"
    first_tool = InterruptingWeigher(interrupt=True)
    first = GoldDensityWeigher(
        table_path, weigher=first_tool, chunk=2, random_state=random_state
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        first.weigh_in_table(dataset)
    assert first_tool.calls == 2
    interrupted_table = pxt.get_table(table_path)
    computed = GoldDensityWeigher.get_weights_per_idx_vector(interrupted_table)
    assert len(computed) == 2
    partial_sample_scores = GoldDensityWeigher.get_weights_per_idx(interrupted_table)
    normalized = GoldDensityWeigher.normalize_weights(partial_sample_scores)
    assert sum(normalized.values()) / len(normalized) == pytest.approx(1.0)
    assert GoldDensityWeigher.get_weights_per_idx_vector(interrupted_table) == computed

    resumed_tool = InterruptingWeigher(interrupt=False)
    resumed = GoldDensityWeigher(
        table_path, weigher=resumed_tool, chunk=2, random_state=random_state
    )
    result = resumed.weigh_in_table(dataset)

    assert resumed_tool.calls == 1
    assert len(resumed.get_weights_per_idx(result)) == 4


@pytest.mark.usefixtures("pxt_unit_test_dir")
@pytest.mark.parametrize("chunk", [None, 1, 100], ids=["full", "single", "oversized"])
def test_table_and_dataset_inputs_preserve_custom_keys_and_sequence_metadata(chunk):
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

    assert table_weigher.get_weights_per_idx(from_table, "density") == pytest.approx(
        dataset_weigher.get_weights_per_idx(from_dataset, "density")
    )
    dataset_view = dataset_weigher.weigh_in_dataset(dataset)
    items = list(dataset_view)
    assert {tuple(item["label_seq"].tolist()) for item in items} == {(2, 3), (5, 8)}
    assert "density_chunk" not in items[0]


@pytest.mark.usefixtures("pxt_unit_test_dir")
@pytest.mark.parametrize("values", [[1.0], [float("nan"), 1.0], [-1.0, 1.0]])
def test_weigher_rejects_malformed_custom_tool_outputs(values):
    dataset = VectorDataset(
        [
            {"idx": 0, "vectorized": torch.tensor([0.0])},
            {"idx": 1, "vectorized": torch.tensor([1.0])},
        ]
    )
    weigher = GoldDensityWeigher(
        "unit_test.bad_tool", weigher=FixedScoreWeigher(values), chunk=None
    )

    with pytest.raises(ValueError, match="one finite nonnegative score"):
        weigher.weigh_in_table(dataset)


@pytest.mark.usefixtures("pxt_unit_test_dir")
@pytest.mark.parametrize("size", [6, 7], ids=["insufficient", "minimum"])
def test_estimation_does_not_change_torch_rng_state(size):
    source = pxt.create_table(
        "unit_test.estimate_vectors",
        source=[
            {"idx": i, "idx_vector": i, "vectorized": [float(i), 1.0]}
            for i in range(size)
        ],
        primary_key="idx_vector",
    )
    weigher = GoldDensityWeigher("unit_test.estimate_weights")
    torch.manual_seed(927)
    before = torch.get_rng_state().clone()

    estimate = weigher.estimate_computation_time(source)

    if size < 7:
        assert estimate is None
    else:
        assert estimate is not None and estimate >= 0
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_distribution_and_constructor_limits_are_validated():
    with pytest.raises(NotImplementedError, match="Distributed processing"):
        GoldDensityWeigher("unit_test.distribute", distribute=True)
    weigher = GoldDensityWeigher("unit_test.distribute")
    with pytest.raises(NotImplementedError, match="Distributed processing"):
        weigher.distribute = True
    with pytest.raises(ValueError, match="positive integer"):
        GoldDensityWeigher("unit_test.invalid_chunk", chunk=0)


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_existing_result_table_must_have_idx_vector_primary_key():
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
