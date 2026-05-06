import pytest
import torch
from torch.utils.data import Dataset, Subset
import pixeltable as pxt

from goldener import (
    GoldDescriptor,
    GoldClusterizer,
    GoldClusteringTool,
    GoldTorchEmbeddingToolConfig,
    GoldTorchEmbeddingTool,
)
from goldener.organize import (
    GoldClusterizedBatchSampler,
    get_indices_per_cluster_for_subset,
)
from goldener.pxt_utils import pxt_torch_dataset_collate_fn
from goldener.vectorize import GoldVectorizer, TensorVectorizer


class DummyDataset(Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.fixture(scope="function")
def descriptor():
    model = DummyModel()
    config = GoldTorchEmbeddingToolConfig(model=model, layers=None)
    return GoldDescriptor(
        table_path="unit_test.descriptor_batcher",
        embedder=GoldTorchEmbeddingTool(config),
        batch_size=2,
        collate_fn=None,
        device=torch.device("cpu"),
        allow_existing=False,
    )


@pytest.fixture(scope="function")
def vectorizer():
    return GoldVectorizer(
        table_path="unit_test.vectorizer_batcher",
        vectorizer=TensorVectorizer(),
        collate_fn=pxt_torch_dataset_collate_fn,
        batch_size=2,
    )


@pytest.fixture(scope="function")
def clusterizer():
    return GoldClusterizer(
        table_path="unit_test.clusterizer_batcher",
        clustering_tool=DummySingleSizeClusteringTool(),
        allow_existing=False,
    )


class DummySingleSizeClusteringTool(GoldClusteringTool):
    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        cluster_assignment = []
        for assignment_idx in range(len(x) // n_clusters):
            cluster_assignment.extend(list(range(n_clusters)))

        return torch.tensor(cluster_assignment)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DummyMultiSizesClusteringTool(GoldClusteringTool):
    def __init__(self, remove: int) -> None:
        self.remove = remove

    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        cluster_assignment = []
        for assignment_idx in range((len(x) - self.remove) // n_clusters):
            cluster_assignment.extend(list(range(n_clusters)))

        cluster_assignment.extend([0] * self.remove)

        return torch.tensor(cluster_assignment)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DummyEmptySizesClusteringTool(GoldClusteringTool):
    def fit(self, x: torch.Tensor, n_clusters: int) -> torch.Tensor:
        cluster_assignment = []
        for assignment_idx in range(len(x) // (n_clusters - 1)):
            cluster_assignment.extend(list(range(n_clusters - 1)))

        return torch.tensor(cluster_assignment)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TestGoldClusterizedBatchSampler:
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def test_simple_usage(self, clusterizer):
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(20)]
        )
        batch_sampler = GoldClusterizedBatchSampler(
            dataset=dataset,
            clusterizer=clusterizer,
            batch_size=5,
            descriptor=None,
            vectorizer=None,
            force_same_size=True,
            shuffle=False,
            generator=None,
        )
        not_shuffled_batches = [batch for batch in batch_sampler]
        assert len(batch_sampler) == 4
        for batch in not_shuffled_batches:
            assert len(batch) == 5

        not_shuffled_indices = [idx for batch in not_shuffled_batches for idx in batch]
        assert not_shuffled_indices == list(range(20))

        batch_sampler.shuffle = True
        shuffled_indices = [idx for batch in batch_sampler for idx in batch]
        assert sorted(not_shuffled_indices) == sorted(shuffled_indices)
        assert shuffled_indices != not_shuffled_indices

        shuffled_again_indices = [idx for batch in batch_sampler for idx in batch]
        assert shuffled_indices != shuffled_again_indices

    def test_multi_size_cluster(self):
        table_path = "unit_test.clusterizer_batcher"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(20)]
        )
        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=DummyMultiSizesClusteringTool(5),
            allow_existing=False,
        )
        batch_sampler = GoldClusterizedBatchSampler(
            dataset=dataset,
            clusterizer=clusterizer,
            batch_size=5,
            descriptor=None,
            vectorizer=None,
            force_same_size=False,
            shuffle=False,
            generator=None,
        )
        not_shuffled_batches = [batch for batch in batch_sampler]
        assert len(batch_sampler) == 8
        for batch in not_shuffled_batches:
            assert len(batch) == 5

        not_shuffled_indices = [idx for batch in not_shuffled_batches for idx in batch]
        assert not_shuffled_indices != list(range(20))

        batch_sampler.shuffle = True
        shuffled_indices = [idx for batch in batch_sampler for idx in batch]
        assert shuffled_indices != not_shuffled_indices

        shuffled_again_indices = [idx for batch in batch_sampler for idx in batch]
        assert shuffled_indices != shuffled_again_indices

    def test_with_descriptor_and_vectorizer(self, descriptor, vectorizer, clusterizer):
        dataset = DummyDataset(
            [
                {
                    "data": torch.rand(
                        4,
                        1,
                    ),
                    "idx": idx,
                }
                for idx in range(20)
            ]
        )
        batch_sampler = GoldClusterizedBatchSampler(
            dataset=dataset,
            clusterizer=clusterizer,
            batch_size=5,
            descriptor=descriptor,
            vectorizer=vectorizer,
            force_same_size=True,
            shuffle=False,
            generator=None,
        )
        not_shuffled_batches = [batch for batch in batch_sampler]
        assert len(batch_sampler) == 4
        for batch in not_shuffled_batches:
            assert len(batch) == 5

        not_shuffled_indices = [idx for batch in not_shuffled_batches for idx in batch]
        assert not_shuffled_indices == list(range(20))

    def test_force_set_size_failure(self):
        table_path = "unit_test.clusterizer_batcher"
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(20)]
        )
        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=DummyMultiSizesClusteringTool(5),
            allow_existing=False,
        )
        with pytest.raises(
            ValueError, match="All the clusters are required to have the same size"
        ):
            GoldClusterizedBatchSampler(
                dataset=dataset,
                clusterizer=clusterizer,
                batch_size=5,
                descriptor=None,
                vectorizer=None,
                force_same_size=True,
                shuffle=False,
                generator=None,
            )

    def test_with_generator(self, clusterizer):
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(20)]
        )
        clusterizer.allow_existing = True
        generator = torch.Generator().manual_seed(42)
        batch_sampler = GoldClusterizedBatchSampler(
            dataset=dataset,
            clusterizer=clusterizer,
            batch_size=5,
            descriptor=None,
            vectorizer=None,
            force_same_size=True,
            shuffle=True,
            generator=generator,
        )
        shuffled_batches_1 = [batch for batch in batch_sampler]

        generator = torch.Generator().manual_seed(42)
        batch_sampler = GoldClusterizedBatchSampler(
            dataset=dataset,
            clusterizer=clusterizer,
            batch_size=5,
            descriptor=None,
            vectorizer=None,
            force_same_size=True,
            shuffle=True,
            generator=generator,
        )
        shuffled_batches_2 = [batch for batch in batch_sampler]

        assert shuffled_batches_1 == shuffled_batches_2

    def test_empty_cluster_failure(self):
        table_path = "unit_test.clusterizer_batcher"
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(20)]
        )
        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=DummyEmptySizesClusteringTool(),
            allow_existing=False,
        )
        with pytest.raises(
            ValueError,
            match="Some clusters are empty. Please check the clusterizer configuration",
        ):
            GoldClusterizedBatchSampler(
                dataset=dataset,
                clusterizer=clusterizer,
                batch_size=5,
                descriptor=None,
                vectorizer=None,
                force_same_size=False,
                shuffle=False,
                generator=None,
            )

    def test_with_subset(self, clusterizer):
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(20)]
        )
        subset = Subset(dataset, [0, 4, 9, 14, 19])
        batch_sampler = GoldClusterizedBatchSampler(
            dataset=subset,
            clusterizer=clusterizer,
            batch_size=5,
            descriptor=None,
            vectorizer=None,
            force_same_size=True,
            shuffle=False,
            generator=None,
        )
        batches = [batch for batch in batch_sampler]
        assert len(batch_sampler) == 1
        for batch in batches:
            assert len(batch) == 5

        batcher_indices = [idx for batch in batches for idx in batch]
        assert batcher_indices == list(range(5))


class TestGetIndicesPerClusterForSubset:
    def test_basic(self):
        result = get_indices_per_cluster_for_subset(
            indices_per_cluster={0: [10, 20], 1: [30]},
            indices_in_subset=[10, 20, 30],
        )
        assert sorted(result[0]) == [0, 1]
        assert sorted(result[1]) == [
            2,
        ]

    def test_multiple_duplicates_in_subset(self):
        result = get_indices_per_cluster_for_subset(
            indices_per_cluster={
                0: [5, 7],
            },
            indices_in_subset=[5, 7, 7, 5],
        )
        assert sorted(result[0]) == [0, 1, 2, 3]

    def test_raises_when_dataset_index_not_in_subset(self):
        with pytest.raises(
            KeyError,
        ):
            get_indices_per_cluster_for_subset(
                indices_per_cluster={
                    0: [5],
                },
                indices_in_subset=[0, 1, 2],
            )

    def test_empty_dataset_indices(self):
        result = get_indices_per_cluster_for_subset(
            indices_per_cluster={
                0: [],
            },
            indices_in_subset=[0, 1, 2],
        )
        assert result[0] == []
