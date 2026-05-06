from logging import getLogger

import torch
from torch.utils.data import Dataset, Sampler, Subset
from torch import Generator
from goldener.describe import GoldDescriptor
from goldener.clusterize import GoldClusterizer
from goldener.torch_utils import shuffle_list, get_subset_indices_for_indices
from goldener.vectorize import GoldVectorizer


logger = getLogger(__name__)


class GoldClusterizedBatchSampler(Sampler):
    """Batch sampler forcing the presence of all clusters in each batch.

    With a random batcher, the content distribution within the batches might vary a lot.
    During the training of a model, this variation can influence badly its convergence
    leading to an underperforming model.

    In this batcher, the data is first clustered in order to gather together the samples showing a similar content.
    Then, every batch is forced to contain at least 1 element of each cluster.

    The randomness is still available in the process of selection among the elements of all clusters.

    Args:
        dataset: dataset to sample from.
        batch_size: batch size specifying the size of each batch. This is as well the number of clusters to create.
        clusterizer: clusterizer to cluster the data into `batch_size` clusters.
        descriptor: optional descriptor to describe the dataset before clusterization.
        vectorizer: optional vectorizer to vectorize the dataset before clusterization.
        force_same_size: if True, all the clusters are required to have the same size.
            If False, the sampler will cycle through the clusters until all the samples are exhausted. It will
            then oversample the smallest clusters.
        shuffle: if True, the order of the samples of all clusters will be shuffled before sampling,
            the batch is shuffled to change the cluster order, and once exhausted a cluster is shuffled again.
            If False, the order of the samples and clusters will be preserved.
        generator: optional generator to manage the random shuffling.
            If None, a new generator will be created with a random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        clusterizer: GoldClusterizer,
        batch_size: int,
        descriptor: GoldDescriptor | None = None,
        vectorizer: GoldVectorizer | None = None,
        force_same_size: bool = False,
        shuffle: bool = True,
        generator: Generator | None = None,
    ):
        self.shuffle = shuffle
        self.generator = generator

        # clusterize the dataset
        logger.info("Computing the clusters for the GoldClusterizedBatchSampler")
        description = (
            dataset if descriptor is None else descriptor.describe_in_table(dataset)
        )
        vectorized = (
            description
            if vectorizer is None
            else vectorizer.vectorize_in_table(description)
        )
        clusterized = clusterizer.cluster_in_table(vectorized, batch_size)
        self._indices_per_cluster: dict[int, set[int] | list[int]] = {
            cluster_idx: clusterizer.get_cluster_indices(
                table=clusterized,
                cluster_idx=cluster_idx,
                cluster_key=clusterizer.cluster_key,
                idx_key="idx",
            )
            for cluster_idx in range(batch_size)
        }
        if isinstance(dataset, Subset):
            self._indices_per_cluster = {
                cluster_idx: get_subset_indices_for_indices(
                    set(cluster_indices), dataset.indices
                )
                for cluster_idx, cluster_indices in self._indices_per_cluster.items()
            }

        # validate the cluster sizes and compute the max cluster size to know how many batches to draw
        cluster_sizes = [len(c) for c in self._indices_per_cluster.values()]
        if force_same_size and len(set(cluster_sizes)) != 1:
            raise ValueError(
                "All the clusters are required to have the same size when `force_same_size=True`"
            )
        if any(cluster_size == 0 for cluster_size in cluster_sizes):
            raise ValueError(
                "Some clusters are empty. Please check the clusterizer configuration."
            )

        self._max_cluster_size = max(cluster_sizes)

    def __iter__(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # order the indices of each cluster and initialize the tracking for the next index to sample for each cluster
        indices_buckets: dict[int, list[int]] = {}  # keep the order of indices
        pointers: dict[int, int] = {}  # to select the next sample to add in the batch
        for cluster_idx, cluster_indices in self._indices_per_cluster.items():
            pool = sorted(
                cluster_indices
            )  # without shuffle the samples are ordered by index
            if self.shuffle:
                pool = shuffle_list(pool, generator)
            indices_buckets[cluster_idx] = pool
            pointers[cluster_idx] = 0

        # draw the samples ensuring all clusters are present in the batch
        for _ in range(self._max_cluster_size):
            batch = []
            for cluster_idx, cluster_list in indices_buckets.items():
                # the next item to add to the batch is the one corresponding to
                # the pointer
                pool = cluster_list
                ptr = pointers[cluster_idx]
                batch.append(pool[ptr])

                # Increment the pointer. If it exceeds the pool size, wrap around (Cycle)
                new_pointer = (ptr + 1) % len(pool)
                pointers[cluster_idx] = new_pointer

                # if the pointer is exhausted, randomize again the corresponding pool
                if new_pointer == 0 and self.shuffle:
                    indices_buckets[cluster_idx] = shuffle_list(cluster_list, generator)

            # the batch is shuffled to obtain different cluster orders across batches
            if self.shuffle:
                batch = shuffle_list(batch, generator)

            yield batch

    def __len__(self):
        # Length is dictated by the largest cluster to ensure full coverage
        return self._max_cluster_size
