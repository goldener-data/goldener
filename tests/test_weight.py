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


class CountingWeigher(GoldKnnDensityWeighingTool):
    def __init__(self):
        super().__init__(k=1)
        self.calls = 0

    def weigh(self, x):
        self.calls += 1
        return super().weigh(x)


class FixedScoreWeigher(GoldDensityWeighingTool):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def weigh(self, x):
        if isinstance(self.values, Exception):
            raise self.values
        return torch.tensor(self.values, dtype=torch.float64)


class InterruptingWeigher(GoldKnnDensityWeighingTool):
    def __init__(self, interrupt):
        super().__init__(k=1)
        self.interrupt = interrupt
        self.calls = 0

    def weigh(self, x):
        self.calls += 1
        if self.interrupt and self.calls == 2:
            raise RuntimeError("simulated interruption")
        return super().weigh(x)


@pytest.fixture
def pxt_unit_test_dir():
    pxt.drop_dir("unit_test", force=True)
    pxt.create_dir("unit_test", if_exists="ignore")
    yield
    pxt.drop_dir("unit_test", force=True)


def test_knn_density_weights_are_mean_one_and_sparse_points_score_higher():
    x = torch.tensor([[0.0], [0.1], [0.2], [5.0]])
    scores = GoldKnnDensityWeighingTool(k=1).weigh(x)

    assert scores.shape == (4,)
    assert scores.dtype == x.dtype
    assert scores.device == x.device
    assert scores.mean().item() == pytest.approx(1.0)
    assert scores[3] > scores[0]


@pytest.mark.parametrize(
    "x,kwargs,expected",
    [
        (torch.empty((0, 3)), {}, (0,)),
        (torch.tensor([[2.0, 3.0]]), {}, (1,)),
        (torch.zeros((4, 2)), {}, (4,)),
        (torch.tensor([[0.0], [1.0], [2.0]]), {"power": 0}, (3,)),
    ],
)
def test_knn_density_weights_degenerate_populations(x, kwargs, expected):
    scores = GoldKnnDensityWeighingTool(**kwargs).weigh(x)

    assert scores.shape == expected
    if len(scores):
        assert torch.equal(scores, torch.ones_like(scores))


def test_knn_density_excludes_only_self_and_query_batching_preserves_results():
    x = torch.tensor([[0.0], [0.0], [3.0], [8.0]])
    one_batch = GoldKnnDensityWeighingTool(k=1, distance_batch_size=20).weigh(x)
    small_batches = GoldKnnDensityWeighingTool(k=1, distance_batch_size=1).weigh(x)

    assert torch.allclose(one_batch, small_batches)
    assert one_batch[0] == one_batch[1]
    assert one_batch[0] > 0
    assert one_batch[2] > one_batch[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": True},
        {"distance_batch_size": 0},
        {"power": -1},
        {"eps": 0},
    ],
)
def test_knn_density_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GoldKnnDensityWeighingTool(**kwargs)


@pytest.mark.parametrize(
    "x",
    [torch.ones(3), torch.empty((2, 0)), torch.tensor([[float("nan")]])],
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
def test_weigher_resumes_new_dataset_batches_and_aggregates_vectors_per_sample():
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
    assert weigher._state_value(partial, "complete") is None
    weigher.max_batches = 0
    with pytest.raises(ValueError, match="incomplete"):
        weigher.weigh_in_dataset(dataset)

    resumed = GoldDensityWeigher(
        "unit_test.resumable_weights",
        weigher=CountingWeigher(),
        chunk=2,
        batch_size=2,
        max_batches=1,
        to_keep_schema={"label": pxt.String},
    )
    result = resumed.weigh_in_table(dataset)

    assert result.count() == 4
    assert resumed._state_value(result, "complete") == "1"
    assert resumed._state_value(result, "finalized") == "1"
    sample_weights = resumed.get_weights_per_idx(result)
    vector_weights = resumed.get_weights_per_idx_vector(result)
    assert set(sample_weights) == {20, 50}
    assert len(vector_weights) == 4
    assert vector_weights[91] == vector_weights[14] == sample_weights[20]
    assert vector_weights[63] == vector_weights[77] == sample_weights[50]
    assert sum(sample_weights.values()) / len(sample_weights) == pytest.approx(1.0)
    assert {row["label"] for row in result.select(result.label).collect()} == {"a", "b"}

    # A new doer with the same configuration reuses the committed density scores.
    new_tool = CountingWeigher()
    again = GoldDensityWeigher(
        "unit_test.resumable_weights",
        weigher=new_tool,
        chunk=2,
        batch_size=2,
        max_batches=1,
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
    weigher = GoldDensityWeigher("unit_test.drop_weights", drop_table=True)

    result = weigher.weigh_in_dataset(dataset)

    assert len(result) == 2
    assert next(iter(result))["weight"] > 0
    with pytest.raises(pxt.Error):
        pxt.get_table("unit_test.drop_weights")


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_empty_table_returns_empty_finalized_result():
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
    assert weigher._state_value(result, "finalized") == "1"
    assert weigher.get_weights_per_idx(result) == {}


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_sample_weight_normalization_is_mean_one_for_unequal_vector_counts():
    dataset = VectorDataset(
        [
            {"idx": 10, "vectorized": torch.tensor([0.0])},
            {"idx": 10, "vectorized": torch.tensor([0.1])},
            {"idx": 20, "vectorized": torch.tensor([10.0])},
        ]
    )
    weigher = GoldDensityWeigher("unit_test.unequal", chunk=None)

    result = weigher.weigh_in_table(dataset)
    weights = weigher.get_weights_per_idx(result)

    assert weights[20] > weights[10]
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_final_sample_aggregation_handles_extreme_finite_scores():
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

    weights = weigher.get_weights_per_idx(weigher.weigh_in_table(dataset))

    assert weights[0] == pytest.approx(4 / 3)
    assert weights[1] == pytest.approx(2 / 3)


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_interrupted_chunk_resume_skips_committed_tool_calls():
    dataset = VectorDataset(
        [
            {"idx": index, "vectorized": torch.tensor([float(index)])}
            for index in range(4)
        ]
    )
    table_path = "unit_test.interrupted"
    first_tool = InterruptingWeigher(interrupt=True)
    first = GoldDensityWeigher(table_path, weigher=first_tool, chunk=2)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        first.weigh_in_table(dataset)
    assert first_tool.calls == 2

    resumed_tool = InterruptingWeigher(interrupt=False)
    resumed = GoldDensityWeigher(table_path, weigher=resumed_tool, chunk=2)
    result = resumed.weigh_in_table(dataset)

    assert resumed_tool.calls == 1
    assert resumed._state_value(result, "finalized") == "1"


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_table_and_dataset_inputs_preserve_custom_keys_and_sequence_metadata():
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
        "weight_key": "novelty",
        "to_keep_schema": {"label_seq": pxt.Array[(2,), pxt.Int]},
        "collate_fn": custom_collate,
        "chunk": None,
    }
    table_weigher = GoldDensityWeigher("unit_test.table_weights", **options)
    dataset_weigher = GoldDensityWeigher("unit_test.dataset_weights", **options)

    from_table = table_weigher.weigh_in_table(source)
    from_dataset = dataset_weigher.weigh_in_table(dataset)

    assert table_weigher.get_weights_per_idx(from_table, "novelty") == pytest.approx(
        dataset_weigher.get_weights_per_idx(from_dataset, "novelty")
    )
    dataset_view = dataset_weigher.weigh_in_dataset(dataset)
    items = list(dataset_view)
    assert {tuple(item["label_seq"].tolist()) for item in items} == {(2, 3), (5, 8)}
    assert "density_raw" not in items[0]


@pytest.mark.usefixtures("pxt_unit_test_dir")
@pytest.mark.parametrize("values", [[1.0], [float("nan"), 1.0]])
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

    with pytest.raises(ValueError, match="one finite positive score"):
        weigher.weigh_in_table(dataset)


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_estimation_does_not_change_torch_rng_state():
    source = pxt.create_table(
        "unit_test.estimate_vectors",
        source=[
            {"idx": 0, "idx_vector": 0, "vectorized": [0.0, 1.0]},
            {"idx": 1, "idx_vector": 1, "vectorized": [1.0, 1.0]},
        ],
        primary_key="idx_vector",
    )
    weigher = GoldDensityWeigher("unit_test.estimate_weights")
    torch.manual_seed(927)
    before = torch.get_rng_state().clone()

    estimate = weigher.estimate_computation_time(source)

    assert estimate is not None and estimate >= 0
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_configuration_source_population_and_existing_table_are_validated():
    dataset = VectorDataset(
        [
            {"idx": 0, "idx_vector": 10, "vectorized": torch.tensor([0.0])},
            {"idx": 1, "idx_vector": 11, "vectorized": torch.tensor([1.0])},
            {"idx": 2, "idx_vector": 12, "vectorized": torch.tensor([2.0])},
        ]
    )
    staged = GoldDensityWeigher("unit_test.population", batch_size=1, max_batches=1)
    staged.weigh_in_table(dataset)
    dataset.samples = [dataset.samples[1], dataset.samples[2]]
    with pytest.raises(ValueError, match="removed staged rows"):
        staged.weigh_in_table(dataset)

    full = GoldDensityWeigher("unit_test.config", chunk=None)
    full.weigh_in_table(dataset)
    changed = GoldDensityWeigher(
        "unit_test.config", weigher=GoldKnnDensityWeighingTool(k=2), chunk=None
    )
    with pytest.raises(ValueError, match="configuration differs"):
        changed.weigh_in_table(dataset)
    with pytest.raises(ValueError, match="already exists"):
        GoldDensityWeigher("unit_test.config", allow_existing=False).weigh_in_table(
            dataset
        )


def test_distribution_and_mutable_chunk_validation():
    with pytest.raises(NotImplementedError, match="Distributed processing"):
        GoldDensityWeigher("unit_test.distribute", distribute=True)
    weigher = GoldDensityWeigher("unit_test.distribute")
    with pytest.raises(NotImplementedError, match="Distributed processing"):
        weigher.distribute = True
    with pytest.raises(ValueError, match="positive integer"):
        weigher.chunk = 0


@pytest.mark.usefixtures("pxt_unit_test_dir")
def test_existing_result_and_state_schemas_are_checked_before_adoption():
    pxt.create_table(
        "unit_test.unpaired",
        source=[
            {
                "idx": 0,
                "idx_vector": 0,
                "vectorized": [0.0],
            }
        ],
        primary_key="idx_vector",
    )
    with pytest.raises(ValueError, match="no companion state"):
        GoldDensityWeigher("unit_test.unpaired").weigh_in_table(VectorDataset([]))

    pxt.create_table(
        "unit_test.wrong_result_type",
        schema={
            "idx": pxt.Required[pxt.Int],
            "idx_vector": pxt.Required[pxt.String],
        },
        primary_key="idx_vector",
    )
    with pytest.raises(ValueError, match="idx_vector.*IntType"):
        GoldDensityWeigher("unit_test.wrong_result_type").weigh_in_table(
            VectorDataset([])
        )

    pxt.create_table(
        "unit_test.bad_state_density_state",
        schema={
            "key": pxt.Required[pxt.String],
            "value": pxt.Required[pxt.String],
        },
        primary_key="value",
    )
    with pytest.raises(ValueError, match="primary key"):
        GoldDensityWeigher("unit_test.bad_state").weigh_in_table(VectorDataset([]))
