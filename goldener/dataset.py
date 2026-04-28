from torch.utils.data import Dataset, Sampler


from goldener.describe import GoldDescriptor
from goldener.clusterize import GoldClusterizer
from goldener.vectorize import GoldVectorizer

import random


class GoldDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        clusterizer: GoldClusterizer,
        batch_size: int,
        descriptor: GoldDescriptor | None = None,
        vectorizer: GoldVectorizer | None = None,
    ) -> None:
        self._dataset = dataset
        self._clusterizer = clusterizer
        self._batch_size = batch_size
        self._descriptor = descriptor
        self._vectorizer = vectorizer

        description = (
            dataset
            if self._descriptor is None
            else self._descriptor.describe_in_table(dataset)
        )
        vectorized = (
            description
            if self._vectorizer is None
            else self._vectorizer.vectorize_in_table(description)
        )
        clusterized = self._clusterizer.cluster_in_table(vectorized, batch_size)
        self._indices_per_cluster = {
            cluster_idx: self._clusterizer.get_cluster_indices(
                table=clusterized,
                cluster_idx=cluster_idx,
                cluster_key=self._clusterizer.cluster_key,
            )
            for cluster_idx in range(self._batch_size)
        }
        self._dataset_size = sum([len(c) for c in self._indices_per_cluster.values()])

    @property
    def indices_per_cluster(self):
        return self._indices_per_cluster

    def __len__(self) -> int:
        return self._dataset_size


class GoldBatchSampler(Sampler):
    def __init__(
        self,
        indices_per_cluster: dict[int, list[int]],
        shuffle: bool = True,
    ):
        self.shuffle = shuffle
        self.indices_per_cluster = indices_per_cluster

        # the biggest cluster fix the number of iterations
        sizes = [len(c) for c in self.indices_per_cluster.values()]
        self.max_cluster_size = max(sizes)

    def __iter__(self):
        indices_buckets: dict[int, list[int]] = {}  # keep the order of indices
        pointers: dict[int, int] = {}  # to select the next sample to add in the batch
        for cluster_idx, cluster_indices in self.indices_per_cluster.items():
            pool = list(cluster_indices)
            if self.shuffle:
                random.shuffle(pool)
            indices_buckets[cluster_idx] = pool
            pointers[cluster_idx] = 0

        for _ in range(self.max_cluster_size):
            batch = []
            for cluster_idx, cluster_indices in indices_buckets.items():
                # the next item to add to the batch is the one corresponding to
                # the pointer
                pool = cluster_indices
                ptr = pointers[cluster_idx]
                batch.append(pool[ptr])

                # Increment pointer. If it exceeds pool size, wrap around (Cycle)
                new_pointer = (ptr + 1) % len(pool)
                pointers[cluster_idx] = new_pointer

                # if the pointer is exhausted, randomize again the corresponding pool
                if new_pointer == 0 and self.shuffle:
                    random.shuffle(cluster_indices)
                    indices_buckets[cluster_idx] = cluster_indices

            if self.shuffle:
                random.shuffle(batch)

            yield batch

    def __len__(self):
        # Length is dictated by the largest cluster to ensure full coverage
        return self.max_cluster_size
