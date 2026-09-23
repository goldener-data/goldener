from functools import partial

import pixeltable as pxt
import pytest
import torch

from goldener.clusterize import GoldClusterizer, GoldRandomClusteringTool
from goldener.describe import GoldDescriptor
from goldener.do import GoldDoer, GoldDoerWithTable
from goldener.embed import GoldTorchEmbeddingTool, GoldTorchEmbeddingToolConfig
from goldener.select import GoldSelector
from goldener.torch_utils import collate_keeping_sequences_as_sequences
from goldener.vectorize import GoldTensorVectorizationTool, GoldVectorizer


@pytest.fixture(
    params=[
        GoldDoer,
        GoldDoerWithTable,
        GoldDescriptor,
        GoldSelector,
        GoldClusterizer,
        GoldVectorizer,
    ],
    ids=lambda cls: cls.__name__,
)
def make_doer(request):
    cls = request.param
    if cls in (GoldDoer, GoldDoerWithTable):
        return cls
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
    return partial(cls, **kwargs)


def test_defaults(make_doer):
    doer = make_doer()

    assert isinstance(doer, GoldDoer)
    assert doer.allow_existing is True
    assert doer.drop_table is False
    assert doer.max_batches is None
    if type(doer) is not GoldDoer:
        assert isinstance(doer, GoldDoerWithTable)
        assert doer.collate_fn is collate_keeping_sequences_as_sequences
        assert doer.to_keep_schema is None
        assert doer.min_pxt_insert_size == 100
        assert doer.batch_size == 1
        assert doer.num_workers == 0


def test_custom_settings(make_doer):
    def collate_fn(batch):
        return batch

    schema = {"label": pxt.String}
    kwargs = {"allow_existing": False, "drop_table": True, "max_batches": 3}
    if make_doer is not GoldDoer:
        kwargs.update(
            collate_fn=collate_fn,
            to_keep_schema=schema,
            min_pxt_insert_size=10,
            batch_size=4,
            num_workers=2,
        )

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
