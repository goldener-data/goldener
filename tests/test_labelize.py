from unittest.mock import Mock

import numpy as np
import pixeltable as pxt
import pytest
import torch

from goldener import GoldKnnLabelPredictionTool, GoldLabelPredictor
from goldener.select import DistanceType


@pytest.fixture
def pxt_unit_test_dir():
    pxt.drop_dir("unit_test", force=True)
    pxt.create_dir("unit_test", if_exists="ignore")
    yield
    pxt.drop_dir("unit_test", force=True)


@pytest.mark.parametrize("distance", list(DistanceType))
@pytest.mark.parametrize("block_size", [1, 10])
def test_neighbor_vote_fractions_and_reference_snapshot(distance, block_size):
    reference = torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [0.0, 1.0]])
    targets = torch.tensor([0, 1, 1, 0])
    tool = GoldKnnLabelPredictionTool(
        k=3, distance=distance, distance_batch_size=block_size
    )
    assert not tool.fitted
    tool.fit(reference, targets)
    reference.fill_(100)
    targets.fill_(0)
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    probabilities = tool.predict(query)
    torch.testing.assert_close(probabilities, query.new_tensor([[1 / 3, 2 / 3]]))
    assert probabilities.dtype == query.dtype
    assert probabilities.device == query.device
    assert tool.predict(query[:0]).shape == (0, 2)
    tool.k = 20
    torch.testing.assert_close(tool.predict(query), query.new_tensor([[0.5, 0.5]]))


@pytest.mark.parametrize(
    "x,target,match",
    [
        (torch.ones(2), torch.tensor([0, 1]), "2D"),
        (torch.ones(0, 2), torch.tensor([], dtype=torch.long), "empty"),
        (torch.ones(2, 0), torch.tensor([0, 1]), "feature width"),
        (torch.tensor([[float("nan")]]), torch.tensor([0]), "finite"),
        (torch.ones(2, 1), torch.tensor([0]), "one integer"),
        (torch.ones(2, 1), torch.tensor([0.0, 1.0]), "one integer"),
        (torch.ones(2, 1), torch.tensor([0, 2]), "consecutive"),
    ],
)
def test_invalid_reference(x, target, match):
    with pytest.raises(ValueError, match=match):
        GoldKnnLabelPredictionTool().fit(x, target)


def test_prediction_requires_fit_and_matching_features():
    tool = GoldKnnLabelPredictionTool()
    with pytest.raises(ValueError, match="Fit"):
        tool.predict(torch.ones(1, 2))
    tool.fit(torch.ones(1, 2), torch.tensor([0]))
    with pytest.raises(ValueError, match="feature_dim"):
        tool.predict(torch.ones(1, 3))


@pytest.mark.parametrize("source_type", ["dataset", "table"])
@pytest.mark.parametrize("retain_vectors", [False, True])
def test_prediction_resume_ids_annotations_and_retrieval(
    pxt_unit_test_dir, source_type, retain_vectors
):
    reference = [
        {"vectorized": torch.tensor([1.0, 0.0]), "label": "cat"},
        {"vectorized": torch.tensor([0.0, 1.0]), "label": "dog"},
    ]
    source = [
        {
            "idx": 10,
            "idx_vector": 3,
            "vectorized": torch.tensor([1.0, 0.0]),
            "label": None,
            "note": "a",
        },
        {
            "idx": 10,
            "idx_vector": 8,
            "vectorized": torch.tensor([0.0, 1.0]),
            "label": None,
            "note": "b",
        },
        {
            "idx": 20,
            "idx_vector": 9,
            "vectorized": torch.tensor([1.0, 0.0]),
            "label": "dog",
            "note": "c",
        },
    ]
    if source_type == "table":
        table = pxt.create_table(
            "unit_test.source",
            {
                "idx": pxt.Int,
                "idx_vector": pxt.Required[pxt.Int],
                "vectorized": pxt.Array[(2,), np.float32],
                "label": pxt.String,
                "note": pxt.String,
            },
            primary_key="idx_vector",
        )
        table.insert(
            [{**row, "vectorized": row["vectorized"].numpy()} for row in source]
        )
        source = table
    tool = GoldKnnLabelPredictionTool(k=1)
    doer = GoldLabelPredictor(
        "unit_test.prediction",
        predictor=tool,
        label_key="label",
        include_vectorized_in_table=retain_vectors,
        to_keep_schema={"note": pxt.String},
        max_batches=1,
        min_pxt_insert_size=1,
    )
    doer.fit(reference)
    tool.predict = Mock(wraps=tool.predict)
    first = doer.predict_in_table(source)
    assert first.count() == 3
    assert len(doer.get_predictions(first)) == 1
    assert tool.predict.call_count == 1
    doer.max_batches = None
    completed = doer.predict_in_table(source)
    assert doer.get_predictions(completed) == {
        3: {"cat": 1.0, "dog": 0.0},
        8: {"cat": 0.0, "dog": 1.0},
        9: {"cat": 0.0, "dog": 1.0},
    }
    assert doer.get_confident_indices(completed, 0.9) == {
        (10, "cat"),
        (10, "dog"),
        (20, "dog"),
    }
    assert doer.get_confident_indices(
        completed, 1, label="cat", idx_key="idx_vector"
    ) == {(3, "cat")}
    assert doer.get_confident_indices(completed, 1, label="unknown") == set()
    assert ("vectorized" in completed.columns()) == (
        source_type == "dataset" or retain_vectors
    )
    assert {r["note"] for r in completed.collect()} == {"a", "b", "c"}
    doer.predict_in_table(source)
    assert tool.predict.call_count == 3
    with pytest.raises(ValueError, match="Cannot refit"):
        doer.fit(reference)
    doer.allow_existing = False
    with pytest.raises(ValueError, match="allow_existing"):
        doer.predict_in_table(source)
    doer.allow_existing = True
    doer.drop_table = True
    dataset = doer.predict_in_dataset(source)
    assert len(dataset) == 3
    assert len(list(dataset)) == 3
    assert pxt.get_table(doer.table_path, if_not_exists="ignore") is None


def test_target_mapping_exclusions_and_table_reference(pxt_unit_test_dir):
    reference = pxt.create_table(
        "unit_test.reference",
        {
            "v": pxt.Array[(2,), np.float32],
            "target": pxt.Array[(2,), np.int64],
        },
    )
    reference.insert(
        [
            {"v": np.array([1, 0], dtype=np.float32), "target": np.array([1, 0])},
            {"v": np.array([0, 1], dtype=np.float32), "target": np.array([0, 1])},
        ]
    )
    doer = GoldLabelPredictor(
        "unit_test.mapping",
        vectorized_key="v",
        prediction_key="p",
        label_key="target",
        target_to_label={(1, 0): "cat", (0, 1): "dog"},
        exclude_labels={"dog"},
    )
    result = doer.predict_in_table([{"v": torch.tensor([0.0, 1.0])}], reference)
    assert doer.classes == ["cat"]
    assert doer.get_predictions(result, "p") == {0: {"cat": 1.0}}
    assert doer.estimate_computation_time([]) is None


@pytest.mark.parametrize(
    "kwargs,match,error",
    [
        ({"distribute": True}, "Distributed", NotImplementedError),
        ({"chunk": 0}, "chunk", ValueError),
        ({"batch_size": True}, "batch_size", ValueError),
        ({"max_batches": -1}, "max_batches", ValueError),
        ({"prediction_key": "idx_vector"}, "identity column", ValueError),
        ({"label_key": "prediction"}, "share the same", ValueError),
        ({"to_keep_schema": {"prediction": pxt.Json}}, "redefine", ValueError),
    ],
)
def test_invalid_configuration(kwargs, match, error):
    with pytest.raises(error, match=match):
        GoldLabelPredictor("unit_test.invalid", **kwargs)


def test_missing_fit_and_mutable_distribution():
    doer = GoldLabelPredictor("unit_test.invalid")
    with pytest.raises(ValueError, match="Fit"):
        doer.predict_in_table([])
    with pytest.raises(NotImplementedError, match="Distributed"):
        doer.distribute = True


@pytest.mark.parametrize("threshold", [-1, 1.1, float("nan")])
def test_invalid_confidence_threshold(threshold):
    with pytest.raises(ValueError, match="min_confidence"):
        GoldLabelPredictor.get_confident_indices(None, threshold)


@pytest.mark.parametrize(
    "probabilities",
    [
        [[1.0]],
        [[-0.1, 1.1]],
        [[0.1, 0.1]],
        [[float("nan"), 0.0]],
    ],
)
def test_custom_tool_invalid_probabilities(pxt_unit_test_dir, probabilities):
    tool = GoldKnnLabelPredictionTool(k=1)
    tool.fit(torch.eye(2), torch.tensor([0, 1]))
    tool.predict = Mock(return_value=torch.tensor(probabilities))
    doer = GoldLabelPredictor(
        "unit_test.invalid_output", predictor=tool, classes=["cat", "dog"]
    )
    with pytest.raises(ValueError, match="one probability per vector and class"):
        doer.predict_in_table([{"vectorized": torch.tensor([1.0, 0.0])}])
    assert doer.get_predictions(pxt.get_table(doer.table_path)) == {}


def test_new_instance_resumes_with_same_reference(pxt_unit_test_dir):
    reference = [
        {"vectorized": torch.tensor([1.0, 0.0]), "label": "cat"},
        {"vectorized": torch.tensor([0.0, 1.0]), "label": "dog"},
    ]
    queries = [
        {"vectorized": torch.tensor([1.0, 0.0])},
        {"vectorized": torch.tensor([0.0, 1.0])},
    ]
    first = GoldLabelPredictor(
        "unit_test.restart", predictor=GoldKnnLabelPredictionTool(k=1), max_batches=1
    )
    first.predict_in_table(queries, reference)
    restored = GoldLabelPredictor(
        "unit_test.restart", predictor=GoldKnnLabelPredictionTool(k=1)
    )
    restored.fit(reference)
    restored.predictor.predict = Mock(wraps=restored.predictor.predict)
    table = restored.predict_in_table(queries)
    assert restored.get_predictions(table) == {
        0: {"cat": 1.0, "dog": 0.0},
        1: {"cat": 0.0, "dog": 1.0},
    }
    assert restored.predictor.predict.call_count == 1


@pytest.mark.parametrize("field", ["k", "distance_batch_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_neighbor_configuration(field, value):
    with pytest.raises(ValueError, match=field):
        GoldKnnLabelPredictionTool(**{field: value})


def test_unanimous_neighbor_votes_are_exact_probabilities():
    tool = GoldKnnLabelPredictionTool(k=10)
    tool.fit(torch.ones(10, 2), torch.zeros(10, dtype=torch.long))
    torch.testing.assert_close(
        tool.predict(torch.ones(1, 2)), torch.ones(1, 1), rtol=0, atol=0
    )
