from enum import Enum
from logging import getLogger
from typing import Sequence

import torch
from torch.utils.data import Dataset, Sampler, Subset
from torch import Generator
from goldener.describe import GoldDescriptor
from goldener.clusterize import GoldClusterizer
from goldener.torch_utils import shuffle_list
from goldener.vectorize import GoldVectorizer


logger = getLogger(__name__)


def get_indices_per_cluster_for_subset(
    indices_per_cluster: dict[int, list[int]],
    indices_in_subset: Sequence[int],
) -> dict[int, list[int]]:
    """Get the indices per cluster corresponding to the subset of a dataset.

    The initial dataset does not have any duplicate indices. A Subset of a dataset is storing the dataset
    and the sequence of indices corresponding to the subset. The same dataset index can be present multiple
    times in the subset. This function is adapted to this situation and include all the subset locations
    in the output list of indices for each cluster.

    Args:
        indices_per_cluster: The list of indices pointing toward samples in the initial dataset for each cluster.
        indices_in_subset: The indices specifying the subset of the initial dataset.

    Returns: The converted indices_per_cluster mapping the cluster with the indices in the subset
    """
    # build a mapping from the dataset indices to their positions in the subset
    dataset_pos_in_subset = {}

    for subset_pos, subset_index in enumerate(indices_in_subset):
        if subset_index not in dataset_pos_in_subset:
            dataset_pos_in_subset[subset_index] = [subset_pos]
        else:
            dataset_pos_in_subset[subset_index].append(subset_pos)

    return {
        cluster_idx: [
            subset_pos
            for cluster_index in cluster_indices
            for subset_pos in dataset_pos_in_subset[cluster_index]
        ]
        for cluster_idx, cluster_indices in indices_per_cluster.items()
    }


class ExhaustedClusterStrategy(Enum):
    """Strategy to adopt when a cluster is exhausted within a clusterized Batch sampler.

    Inside the batch sampler with clustering, the clusters might be of different size. Thus, some clusters might be
    exhausted before the others. Here the different strategy for the batch sampler:
    -  RESTART: The exhausted cluster is reinitialized and still used for next batch sampling.
        This means that the samples of the exhausted cluster will be oversampled compared to the other clusters.
    - STOP: The batch sampler is stopped when a cluster is exhausted.
        This means that not all the samples of the other clusters will be used.
    - EXCLUDE: The exhausted cluster is excluded from the next batch sampling. This means that the samples
        of the exhausted cluster will be used only once, and that the batches will contain less
        and less clusters until all the clusters are exhausted.


    """

    RESTART = "restart"
    STOP = "stop"
    EXCLUDE = "exclude"


class GoldClusterizedBatchSampler(Sampler):
    """Batch sampler forcing the presence of all clusters in each batch.

    With a random batcher, the content distribution within the batches might vary a lot.
    During the training of a model, this variation can influence badly its convergence
    leading to an underperforming model.

    In this batcher, the data is first clustered in order to gather together the samples showing a similar content.
    Then, every batch is forced to contain at least 1 element of each cluster.

    The randomness is still available in the process of selection among the elements of all clusters.

    Attributes:
        shuffle: if True, the order of the samples of all clusters will be shuffled before sampling,
            the batch is shuffled to change the cluster order, and once exhausted a cluster is shuffled again.
            If False, the order of the samples and clusters will be preserved.
        generator: optional generator to manage the random shuffling.
            If None, a new generator will be created with a random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        clusterizer: GoldClusterizer,
        n_clusters: int | None = None,
        descriptor: GoldDescriptor | None = None,
        vectorizer: GoldVectorizer | None = None,
        force_same_size: bool = False,
        shuffle: bool = True,
        generator: Generator | None = None,
        strategy: ExhaustedClusterStrategy = ExhaustedClusterStrategy.RESTART,
    ):
        """Initialize the batch sampler based on clustering results.

        Args:
            dataset: dataset to sample from.
            batch_size: batch size specifying the size of each batch. Fix the number of clusters if
                n_clusters is None.
            clusterizer: clusterizer to cluster the data into clusters.
            n_clusters: Number of clusters to create. If None, it is set to the batch size.
                It can be different from the batch size if we want to have more clusters
                than the batch size, and thus not all the clusters are present in each batch.
            descriptor: optional descriptor to describe the dataset before clusterization.
            vectorizer: optional vectorizer to vectorize the dataset before clusterization.
            force_same_size: if True, all the clusters are required to have the same size.
                If False, the sampler will cycle through the clusters until all the samples are exhausted. It will
                then oversample the smallest clusters.
            shuffle: if True, the order of the samples of all clusters will be shuffled before sampling,
                and once exhausted a cluster is shuffled again. If False, the order of the samples
                and clusters will be preserved.
            generator: optional generator to manage the random shuffling.
                If None, a new generator will be created with a random seed.
            strategy: strategy to apply when a cluster is exhausted. See ExhaustedClusterStrategy for more details.
        """
        self.shuffle = shuffle
        self.generator = generator

        self._batch_size = batch_size
        self._strategy = strategy

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
        if n_clusters is None:
            n_clusters = batch_size
        clusterized = clusterizer.cluster_in_table(vectorized, n_clusters)
        if clusterized.count() == 0:
            raise ValueError("No samples are present in the dataset.")

        indices_per_cluster = {
            cluster_idx: list(
                clusterizer.get_cluster_indices(
                    table=clusterized,
                    cluster_idx=cluster_idx,
                    cluster_key=clusterizer.cluster_key,
                    idx_key="idx",
                )
            )
            for cluster_idx in range(n_clusters)
        }
        if isinstance(dataset, Subset):
            indices_per_cluster = get_indices_per_cluster_for_subset(
                indices_per_cluster,
                dataset.indices,
            )

        self._indices_per_cluster = {
            cluster_idx: cluster_indices
            for cluster_idx, cluster_indices in indices_per_cluster.items()
            if len(cluster_indices) > 0
        }

        cluster_sizes = [len(c) for c in self._indices_per_cluster.values()]
        if force_same_size and len(set(cluster_sizes)) != 1:
            raise ValueError(
                "All the clusters are required to have the same size when `force_same_size=True`"
            )

    def __iter__(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # order the indices of each cluster and initialize the tracking for the next index to sample for each cluster
        cluster_pools: dict[int, list[int]] = {}  # to set the ordering for each cluster
        cluster_pointers: dict[
            int, int
        ] = {}  # to select the next sample to add in the batch for each cluster
        remaining_clusters: list[
            int
        ] = []  # to track the clusters that are not exhausted yet
        exhausted_once: set[int] = set()  # to stop batching
        sampled_clusters: set[int] = set()
        for cluster_idx, cluster_indices in self._indices_per_cluster.items():
            pool = sorted(
                cluster_indices
            )  # without shuffle the samples are ordered by index
            if self.shuffle:
                pool = shuffle_list(pool, generator)
            cluster_pools[cluster_idx] = pool
            cluster_pointers[cluster_idx] = 0
            remaining_clusters.append(cluster_idx)

        # define the drawing process
        def draw_samples(
            to_sample_from: list[int],
            to_sample_size: int,
        ):
            """Closure drawing the next samples for the batch (one per remaining clusters).

            It defines from which cluster the samples are drawn. The samples are drawn from the pointer status of
            every cluster and this pointer is updated in place. Depending on the exhaustion strategy,
            the exhausted clusters can as well be shuffled in place.
            """
            assert to_sample_size <= len(to_sample_from)

            if to_sample_size < len(to_sample_from):
                # start by sampling in clusters not yet in the current sampled ones
                prioritized_to_sample_from = [
                    cluster_idx
                    for cluster_idx in to_sample_from
                    if cluster_idx not in sampled_clusters
                ]

                if len(prioritized_to_sample_from) > to_sample_size:
                    # remove clusters if still too many samples
                    if self.shuffle:
                        prioritized_to_sample_from = shuffle_list(
                            items=prioritized_to_sample_from, generator=generator
                        )
                    prioritized_to_sample_from = prioritized_to_sample_from[
                        :to_sample_size
                    ]
                elif len(prioritized_to_sample_from) < to_sample_size:
                    # if not enough samples take from already sampled cluster
                    still_to_sample_size = to_sample_size - len(
                        prioritized_to_sample_from
                    )
                    already_sampled_to_sample_from = [
                        cluster_idx
                        for cluster_idx in to_sample_from
                        if cluster_idx in sampled_clusters
                    ]
                    if still_to_sample_size < len(already_sampled_to_sample_from):
                        if self.shuffle:
                            already_sampled_to_sample_from = shuffle_list(
                                items=already_sampled_to_sample_from,
                                generator=generator,
                            )
                        already_sampled_to_sample_from = already_sampled_to_sample_from[
                            :still_to_sample_size
                        ]
                    prioritized_to_sample_from.extend(already_sampled_to_sample_from)

                to_sample_from = prioritized_to_sample_from

            sampled_clusters.update(to_sample_from)
            cluster_samples = {
                cluster_idx: cluster_pool
                for cluster_idx, cluster_pool in cluster_pools.items()
                if cluster_idx in to_sample_from
            }

            samples = []

            for cluster_idx, cluster_pool in cluster_samples.items():
                # the next item to add to the batch is the one corresponding to the pointer
                ptr = cluster_pointers[cluster_idx]
                samples.append(cluster_pool[ptr])

                # the pointer is updated depending on the exhaustion strategy
                new_pointer = ptr + 1
                if self._strategy is ExhaustedClusterStrategy.RESTART:
                    new_pointer = new_pointer % len(cluster_pool)

                    # if the pointer is exhausted, randomize again the corresponding pool
                    if new_pointer == 0:
                        exhausted_once.add(cluster_idx)
                        if self.shuffle:
                            cluster_pools[cluster_idx] = shuffle_list(
                                cluster_pool, generator
                            )
                else:
                    if new_pointer >= len(cluster_pool):
                        # Stop and remove strategies are based on -1 value for exhausted clustered
                        new_pointer = -1
                        exhausted_once.add(cluster_idx)

                cluster_pointers[cluster_idx] = new_pointer

            return samples

        # draw the different batches successively
        # all the samples must be drawn at least once
        # number of batches depends on the batch size, cluster number and composition, and the exhaustion strategy
        while len(exhausted_once) < len(cluster_pools):
            if self._batch_size > len(remaining_clusters):
                # requires to select multiple times from remaining clusters
                batch = []
                while len(batch) < self._batch_size and len(remaining_clusters) > 0:
                    still_to_get = self._batch_size - len(batch)
                    batch.extend(
                        draw_samples(
                            to_sample_from=remaining_clusters,
                            to_sample_size=min(still_to_get, len(remaining_clusters)),
                        )
                    )

                    remaining_clusters = [
                        cluster_idx
                        for cluster_idx, ptr in cluster_pointers.items()
                        if ptr != -1
                    ]
            else:
                # get all samples at once
                batch = draw_samples(
                    to_sample_from=remaining_clusters,
                    to_sample_size=self._batch_size,
                )

            if sampled_clusters.issuperset(set(remaining_clusters)):
                sampled_clusters = set()

            yield batch

            if self._strategy is ExhaustedClusterStrategy.STOP and any(
                ptr == -1 for ptr in cluster_pointers.values()
            ):
                # if the strategy is to stop at exhaustion, the batch sampling is stopped when a cluster is exhausted
                break

            remaining_clusters = [
                cluster_idx
                for cluster_idx, ptr in cluster_pointers.items()
                if ptr != -1
            ]
