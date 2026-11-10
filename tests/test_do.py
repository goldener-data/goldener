from functools import partial

import pixeltable as pxt
import pytest
import torch

from goldener.clusterize import GoldClusterizer, GoldRandomClusteringTool
from goldener.describe import GoldDescriptor
from goldener.do import GoldDoer, GoldDoerWithTable
from goldener.embed import GoldTorchEmbeddingTool, GoldTorchEmbeddingToolConfig
from goldener.novelty import GoldNoveltyScorer
from goldener.select import GoldGreedyKCenterSelectionTool, GoldSelector
from goldener.novelty import GoldNoveltyScorer
from goldener.torch_utils import collate_keeping_sequences_as_sequences
from goldener.vectorize import GoldTensorVectorizationTool, GoldVectorizer
from goldener.weigh import GoldDensityWeigher


@pytest.fixture(
    params=[
        GoldDoer,
        GoldDoerWithTable,
        GoldDescriptor,
        GoldSelector,
        GoldClusterizer,
        GoldVectorizer,
        GoldNoveltyScorer,
        GoldDensityWeigher,
    ],
    ids=lambda cls: cls.__name__,
)
def make_doer(request):
    if request.param is GoldDoer:
        return GoldDoer
    return _table_doer(request.param)


def _table_doer(cls):
    kwargs = {"table_path": "unit_test.doer"}
    if cls is GoldDescriptor:
        kwargs["embedder"] = GoldTorchEmbeddingTool(
            GoldTorchEmbeddingToolConfig(model=torch.nn.Identity())
        )
        kwargs["device"] = torch.device("cpu")
    elif cls is GoldClusterizer:
        kwargs["clustering_tool"] = GoldRandomClusteringTool()
    elif cls is GoldVectorizer:
        kwargs["vectorizer"] = GoldTensorVectorizationTool()
    elif cls is GoldSelector:
        kwargs["selection_tool"] = GoldGreedyKCenterSelectionTool(device="cpu")
    return partial(cls, **kwargs)


def test_defaults(make_doer):
    doer = make_doer()

    assert isinstance(doer, GoldDoer)
    assert doer.allow_existing is True
    assert doer.drop_table is False
    assert doer.max_batches is None
    if isinstance(doer, (GoldSelector, GoldDensityWeigher, GoldNoveltyScorer)):
        assert doer.include_vectorized_in_table is False
    if isinstance(doer, GoldDoerWithTable):
        assert doer.collate_fn is collate_keeping_sequences_as_sequences
        assert doer.to_keep_schema is None
        assert doer.min_pxt_insert_size == 100
        assert doer.batch_size == 1
        assert doer.num_workers == 0
        assert doer.table_path == "unit_test.doer"
        assert doer.label_key is None
        assert doer._MINIMAL_SCHEMA["idx"] == pxt.Required[pxt.Int]


def test_custom_settings(make_doer):
    def collate_fn(batch):
        return batch

    schema = {"label": pxt.String}
    kwargs = {"allow_existing": False, "drop_table": True, "max_batches": 3}
    label_key = None
    if make_doer is not GoldDoer:
        kwargs.update(
            collate_fn=collate_fn,
            to_keep_schema=schema,
            min_pxt_insert_size=10,
            batch_size=4,
            num_workers=2,
        )
        if make_doer.func is not GoldNoveltyScorer:
            label_key = "labels"
            kwargs["label_key"] = label_key

    doer = make_doer(**kwargs)

    assert doer.allow_existing is False
    assert doer.drop_table is True
    assert doer.max_batches == 3
    if make_doer is not GoldDoer:
        assert doer.collate_fn is collate_fn
        assert doer.to_keep_schema is schema
        assert doer.min_pxt_insert_size == 10
        assert doer.batch_size == 4
        assert doer.num_workers == 2
        assert doer.table_path == "unit_test.doer"
        assert doer.label_key == label_key


class _DoerWithKeys(GoldDoerWithTable):
    def __init__(
        self,
        data_key: str | None = "data",
        target_key: str | None = "target",
        output_key: str | None = "output",
        **kwargs,
    ) -> None:
        self.data_key = data_key
        self.target_key = target_key
        self.output_key = output_key
        super().__init__(table_path="unit_test.doer", **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param(
            {"label_key": "idx"},
            r"label_key='idx' collides with the identity column 'idx'",
            id="key_is_identity",
        ),
        pytest.param(
            {"data_key": "a", "output_key": "a", "target_key": "b", "label_key": "b"},
            r"share the same column name: 'a' \(data_key, output_key\); "
            r"'b' \(target_key, label_key\)",
            id="all_duplicates_reported",
        ),
        pytest.param(
            {"to_keep_schema": {"idx": pxt.Int, "label": pxt.String}},
            r"cannot redefine the identity columns \['idx'\]",
            id="kept_identity",
        ),
        pytest.param(
            {"output_key": "idx_vector"},
            r"output_key='idx_vector' collides with the identity column "
            r"'idx_vector'\.",
            id="reserved_idx_vector_merged_with_schema",
        ),
        pytest.param(
            {"label_key": "idx_vector"},
            r"label_key='idx_vector' collides with the identity column "
            r"'idx_vector'\.",
            id="reserved_label_key_assigned_by_table_doer",
        ),
        pytest.param(
            {"to_keep_schema": {"idx_vector": pxt.Int, "label": pxt.String}},
            r"cannot redefine the identity columns \['idx_vector'\]",
            id="kept_reserved_idx_vector",
        ),
    ],
)
def test_rejects_column_name_collisions(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _DoerWithKeys(**kwargs)


def test_accepts_kept_column_named_after_key_and_unset_keys():
    doer = _DoerWithKeys(
        target_key=None,
        output_key=None,
        label_key="label",
        to_keep_schema={"label": pxt.String},
    )

    assert doer.label_key == "label"


def test_plain_doer_accepts_index_names():
    class _PlainDoer(GoldDoer):
        def __init__(self, data_key: str) -> None:
            self.data_key = data_key
            super().__init__()

    for value in ("idx", "idx_vector"):
        plain = _PlainDoer(value)
        assert plain.data_key == value
    assert not hasattr(GoldDoer, "RESERVED_KEYS")
    assert not hasattr(GoldDoer, "_RESERVED_KEYS")
    assert not hasattr(GoldDoer, "_key_attributes")
    assert not hasattr(GoldDoer, "_identity_column_names")
    assert not hasattr(GoldDoer, "_validate_key_attributes")


def test_table_doer_merges_reserved_keys_with_schema():
    assert GoldDoerWithTable._RESERVED_KEYS == frozenset({"idx", "idx_vector"})
    doer = GoldDoerWithTable(table_path="unit_test.doer")
    assert doer._identity_column_names() == {"idx", "idx_vector"}
    descriptor = _table_doer(GoldDescriptor)()
    assert descriptor._identity_column_names() == {"idx", "idx_vector"}


_IDENTITY_IDX_VECTOR = (
    r"label_key='idx_vector' collides with the identity column 'idx_vector'\."
)


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(GoldDoerWithTable, id="GoldDoerWithTable"),
        pytest.param(GoldDescriptor, id="GoldDescriptor"),
        pytest.param(GoldSelector, id="GoldSelector"),
        pytest.param(GoldClusterizer, id="GoldClusterizer"),
        pytest.param(GoldVectorizer, id="GoldVectorizer"),
        pytest.param(GoldDensityWeigher, id="GoldDensityWeigher"),
    ],
)
def test_rejects_idx_vector_label_on_each_table_doer(cls):
    with pytest.raises(ValueError, match=_IDENTITY_IDX_VECTOR):
        _table_doer(cls)(label_key="idx_vector")
