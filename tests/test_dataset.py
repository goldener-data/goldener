import pytest
import torch
from torch.utils.data import Dataset
import pixeltable as pxt

from goldener import (
    GoldDescriptor,
    GoldSelector,
    GoldRandomClusteringTool,
    GoldClusterizer,
)
from goldener.dataset import GoldBatchSampler
from goldener.pxt_utils import pxt_torch_dataset_collate_fn
from goldener.vectorize import GoldVectorizer, TensorVectorizer


class DummyDataset(Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


@pytest.fixture(scope="function")
def descriptor(embedder):
    return GoldDescriptor(
        table_path="unit_test.descriptor_split",
        embedder=embedder,
        to_keep_schema={"label": pxt.String},
        batch_size=2,
        collate_fn=None,
        device=torch.device("cpu"),
        allow_existing=False,
    )


@pytest.fixture(scope="function")
def selector():
    return GoldSelector(
        table_path="unit_test.selector_split",
        to_keep_schema={"label": pxt.String},
        vectorized_key="vectorized",
        batch_size=2,
    )


@pytest.fixture(scope="function")
def vectorizer():
    return GoldVectorizer(
        table_path="unit_test.vectorizer_split",
        vectorizer=TensorVectorizer(),
        to_keep_schema={"label": pxt.String},
        collate_fn=pxt_torch_dataset_collate_fn,
        batch_size=2,
    )


clusterizer = GoldClusterizer(
    table_path="unit_test.clusterizer_split",
    clustering_tool=GoldRandomClusteringTool(random_state=0),
    label_key="label",
    allow_existing=True,
)


class TestGoldBatchSampler:
    def test_simple_usage(self):
        indices_per_cluster = {
            cluster_idx: list(range(cluster_idx * 10, cluster_idx * 10 + 10))
            for cluster_idx in range(10)
        }
        batch_sampler = GoldBatchSampler(indices_per_cluster, shuffle=False)
        not_shuffled = [idx for batch in batch_sampler for idx in batch]

        batch_sampler.shuffle = True
        shuffled = [idx for batch in batch_sampler for idx in batch]
        assert shuffled != not_shuffled

        shuffled_again = [idx for batch in batch_sampler for idx in batch]
        assert shuffled != shuffled_again

    def test_with_multisize_cluster(self):
        indices_per_cluster = {
            0: list(range(0, 3)),
            1: list(range(3, 5)),
            2: list(range(5, 6)),
        }
        batch_sampler = GoldBatchSampler(indices_per_cluster, shuffle=False)
        not_shuffled = [idx for batch in batch_sampler for idx in batch]
        assert not_shuffled == [0, 3, 5, 1, 4, 5, 2, 3, 5]

    def test_with_dataloader(self):
        indices_per_cluster = {
            0: list(range(0, 3)),
            1: list(range(3, 5)),
            2: list(range(5, 6)),
        }
        batch_sampler = GoldBatchSampler(indices_per_cluster, shuffle=False)
        dataset = DummyDataset(list(range(6)))

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
        )
        not_shuffled = [idx for batch in iter(dataloader) for idx in batch]
        assert not_shuffled == [0, 3, 5, 1, 4, 5, 2, 3, 5]

        batch_sampler.shuffle = True
        shuffled = [idx for batch in iter(dataloader) for idx in batch]
        assert shuffled != not_shuffled
        shuffled_again = [idx for batch in iter(dataloader) for idx in batch]
        assert shuffled != shuffled_again
