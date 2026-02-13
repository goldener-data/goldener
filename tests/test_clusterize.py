import pytest
import torch
import pixeltable as pxt

from torch.utils.data import Dataset

from goldener.clusterize import GoldRandomClusteringTool, GoldClusterizer
from goldener.reduce import GoldReducer
from goldener.pxt_utils import GoldPxtTorchDataset


class DummyDataset(Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return (
            self._samples[idx].copy()
            if isinstance(self._samples[idx], dict)
            else self._samples[idx]
        )


class TestGoldRandomClusteringTool:
    def test_fit(self):
        n_clusters = 2
        total = 10
        tool = GoldRandomClusteringTool(42)
        clusters = tool.fit(torch.randn(total, 3), n_clusters)
        assert clusters.shape == (total,)
        assert set(clusters.tolist()) == {0, 1}
        assert len([c for c in clusters if c == 0]) == total / n_clusters
        assert not torch.equal(
            clusters, torch.tensor([i % n_clusters for i in range(total)])
        )

    def test_predict(self):
        with pytest.raises(NotImplementedError):
            GoldRandomClusteringTool(42).predict(torch.randn(10, 3))

    def test_with_nclusters_greater_than_total(self):
        n_clusters = 15
        total = 10
        tool = GoldRandomClusteringTool(42)
        with pytest.raises(
            ValueError, match="cannot be greater than the number of samples"
        ):
            tool.fit(torch.randn(total, 3), n_clusters)


class TestGoldClusterizer:
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def _make_src_table(self, path: str, n: int = 10, with_class: bool = False):
        source_rows = []
        for i in range(n):
            row = {
                "idx": i,
                "idx_vector": i,
                "vectorized": torch.rand(4).numpy(),
            }
            if with_class:
                row["label"] = str(i % 2)
            source_rows.append(row)
        return pxt.create_table(path, source=source_rows, if_exists="replace_force")

    def test_cluster_table_creation_from_table(self):
        src_path = "unit_test.src_cluster_table_input"
        cluster_path = "unit_test.test_cluster_from_table"

        src_table = self._make_src_table(src_path, n=5)

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=False,
        )

        cluster_table = clusterizer._cluster_table_from_table(
            cluster_from=src_table,
            old_cluster_table=None,
        )

        assert set(cluster_table.columns()) >= {
            "idx",
            "idx_vector",
            clusterizer.cluster_key,
        }
        row_indices = [
            row["idx_vector"]
            for row in cluster_table.select(cluster_table.idx_vector).collect()
        ]
        assert set(row_indices) == set(range(5))

    def test_cluster_table_from_table_when_missing_vectorized(self):
        src_path = "unit_test.src_cluster_table_invalid"
        src_table = pxt.create_table(
            src_path,
            source=[{"idx": 0, "idx_vector": 0, "notvec": [1, 2, 3]}],
            if_exists="replace_force",
        )

        clusterizer = GoldClusterizer(
            table_path="unit_test.test_cluster",
            clustering_tool=GoldRandomClusteringTool(),
        )

        with pytest.raises(ValueError, match="does not contain the required column"):
            clusterizer._cluster_table_from_table(
                cluster_from=src_table,
                old_cluster_table=None,
            )

    def test_cluster_table_from_dataset(self):
        table_path = "unit_test.test_cluster_initialize"

        sample = {"vectorized": torch.rand(4), "idx": 0}
        dataset = DummyDataset([sample, sample])

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=False,
        )

        cluster_table = clusterizer._cluster_table_from_dataset(
            cluster_from=dataset,
            old_cluster_table=None,
        )

        assert set(cluster_table.columns()) >= {
            clusterizer.vectorized_key,
            clusterizer.cluster_key,
            "idx",
            "idx_vector",
        }
        row_indices = [
            row["idx_vector"]
            for row in cluster_table.select(cluster_table.idx_vector).collect()
        ]
        assert set(row_indices) == {0, 1}

    def test_cluster_in_table_from_dataset(self):
        table_path = "unit_test.test_cluster_from_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(20)]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=5,
        )

        cluster_table = clusterizer.cluster_in_table(dataset, n_clusters=4)

        assert cluster_table.count() == 20
        assert (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
            == 20
        )
        distinct_clusters = [
            row[clusterizer.cluster_key]
            for row in cluster_table.select(cluster_table[clusterizer.cluster_key])
            .distinct()
            .collect()
        ]
        assert set(distinct_clusters).issubset(set(range(4)))

    def test_cluster_in_table_from_table(self):
        src_path = "unit_test.src_cluster_table"
        cluster_path = "unit_test.test_cluster_from_table_full"

        src_table = self._make_src_table(src_path, n=20)

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
        )

        cluster_table = clusterizer.cluster_in_table(src_table, n_clusters=4)

        assert cluster_table.count() == 20
        assert (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
            == 20
        )
        distinct_clusters = [
            row[clusterizer.cluster_key]
            for row in cluster_table.select(cluster_table[clusterizer.cluster_key])
            .distinct()
            .collect()
        ]
        assert set(distinct_clusters).issubset(set(range(4)))

    def test_cluster_in_table_from_table_with_vectorized_included(self):
        src_path = "unit_test.src_cluster_table"
        cluster_path = "unit_test.test_cluster_from_table_full"

        src_table = self._make_src_table(src_path, n=20)

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            include_vectorized_in_table=True,
        )

        cluster_table = clusterizer.cluster_in_table(src_table, n_clusters=4)

        assert cluster_table.count() == 20
        assert (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
            == 20
        )
        distinct_clusters = [
            row[clusterizer.cluster_key]
            for row in cluster_table.select(cluster_table[clusterizer.cluster_key])
            .distinct()
            .collect()
        ]
        assert set(distinct_clusters).issubset(set(range(4)))

    def test_cluster_in_table_with_invalid_n_clusters(self):
        table_path = "unit_test.test_cluster_invalid_n"
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(10)]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
        )

        with pytest.raises(ValueError, match="n_clusters to be greater than 1"):
            clusterizer.cluster_in_table(dataset, n_clusters=1)

    def test_cluster_in_table_with_chunk(self):
        table_path = "unit_test.test_cluster_chunk"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(50)]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=10,
            chunk=10,
        )

        cluster_table = clusterizer.cluster_in_table(dataset, n_clusters=5)

        assert cluster_table.count() == 50
        assert (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
            == 50
        )
        distinct_clusters = [
            row[clusterizer.cluster_key]
            for row in cluster_table.select(cluster_table[clusterizer.cluster_key])
            .distinct()
            .collect()
        ]
        assert set(distinct_clusters).issubset(set(range(5)))

    def test_cluster_in_table_with_chunk_larger_than_total(self):
        table_path = "unit_test.test_cluster_chunk_large"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(15)]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=5,
            chunk=100,  # larger than total number of vectors (15)
        )

        cluster_table = clusterizer.cluster_in_table(dataset, n_clusters=3)

        assert cluster_table.count() == 15
        clustered = (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
        )
        assert clustered == 15

        distinct_clusters = [
            row[clusterizer.cluster_key]
            for row in cluster_table.select(cluster_table[clusterizer.cluster_key])
            .distinct()
            .collect()
        ]
        assert set(distinct_clusters).issubset(set(range(3)))

    def test_cluster_in_table_with_reducer(self):
        table_path = "unit_test.test_cluster_reducer"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(8), "idx": idx} for idx in range(30)]
        )

        from sklearn.decomposition import PCA

        reducer = GoldReducer(PCA(n_components=4))

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=10,
            reducer=reducer,
        )

        cluster_table = clusterizer.cluster_in_table(dataset, n_clusters=3)

        assert cluster_table.count() == 30
        assert (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
            == 30
        )
        distinct_clusters = [
            row[clusterizer.cluster_key]
            for row in cluster_table.select(cluster_table[clusterizer.cluster_key])
            .distinct()
            .collect()
        ]
        assert set(distinct_clusters).issubset(set(range(3)))

    def test_cluster_in_dataset_and_drop_table(self):
        table_path = "unit_test.test_cluster_dataset_drop"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(10)]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=5,
            drop_table=True,
        )

        clustered_dataset = clusterizer.cluster_in_dataset(dataset, n_clusters=2)

        assert isinstance(clustered_dataset, GoldPxtTorchDataset)

        first_item = next(iter(clustered_dataset))
        assert isinstance(first_item, dict)
        assert "idx" in first_item

        with pytest.raises(pxt.Error):
            pxt.get_table(table_path)

    def test_cluster_in_table_with_existing_table_and_disallow(self):
        src_path = "unit_test.src_cluster_table_existing"
        cluster_path = "unit_test.test_cluster_existing"

        src_table = self._make_src_table(src_path, n=5)

        pxt.create_table(
            cluster_path,
            source=[{"idx": 0, "idx_vector": 0, "cluster": 0}],
            if_exists="replace_force",
        )

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=False,
        )

        with pytest.raises(
            ValueError, match="does not have all specified columns as primary keys"
        ):
            clusterizer.cluster_in_table(src_table, n_clusters=2)

    def test_cluster_in_table_with_restart_allowed(self):
        table_path = "unit_test.test_cluster_restart_allowed"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(30)]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=False,
            batch_size=5,
        )

        cluster_table_1 = clusterizer.cluster_in_table(dataset, n_clusters=3)

        assert cluster_table_1.count() == 30
        clustered_count_1 = (
            cluster_table_1.where(
                cluster_table_1[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table_1.idx)
            .distinct()
            .count()
        )
        assert clustered_count_1 == 30

        clusterizer_restart = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=5,
        )

        cluster_table_2 = clusterizer_restart.cluster_in_table(dataset, n_clusters=3)

        assert cluster_table_2.count() == 30
        clustered_count_2 = (
            cluster_table_2.where(
                cluster_table_2[clusterizer_restart.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table_2.idx)
            .distinct()
            .count()
        )
        assert clustered_count_2 == 30
        assert clustered_count_2 == clustered_count_1

    def test_cluster_in_table_with_class_key(self):
        table_path = "unit_test.test_cluster_with_class"

        dataset = DummyDataset(
            [
                {
                    "vectorized": torch.rand(4),
                    "idx": idx,
                    "label": str(idx % 2),
                }
                for idx in range(40)
            ]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=8,
            class_key="label",
        )

        cluster_table = clusterizer.cluster_in_table(dataset, n_clusters=4)

        assert cluster_table.count() == 40
        assert (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
            == 40
        )

        label_col = cluster_table[clusterizer.class_key]
        total_class0 = (
            cluster_table.where(label_col == "0")
            .select(cluster_table.idx)
            .distinct()
            .count()
        )
        total_class1 = (
            cluster_table.where(label_col == "1")
            .select(cluster_table.idx)
            .distinct()
            .count()
        )
        assert total_class0 == 20
        assert total_class1 == 20

        all_indices = clusterizer.get_cluster_vector_indices(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
        )
        assert all_indices == set(range(40))

        class0_indices = clusterizer.get_cluster_vector_indices(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            class_key=clusterizer.class_key,
            class_value="0",
        )
        class1_indices = clusterizer.get_cluster_vector_indices(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            class_key=clusterizer.class_key,
            class_value="1",
        )

        assert class0_indices.union(class1_indices) == set(range(40))
        assert class0_indices.isdisjoint(class1_indices)

    def test_cluster_in_table_with_max_batches(self):
        table_path = "unit_test.test_cluster_max_batches"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(50)]
        )

        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=10,
            max_batches=2,
        )

        cluster_table = clusterizer.cluster_in_table(dataset, n_clusters=3)

        clustered = (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .count()
        )
        assert clustered == 20

    def test_get_cluster_vector_indices(self):
        src_path = "unit_test.test_cluster_indices_explicit_src"
        cluster_path = "unit_test.test_cluster_indices_explicit"

        src_table = self._make_src_table(src_path, n=6, with_class=True)

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            class_key="label",
        )

        cluster_table = clusterizer._cluster_table_from_table(
            cluster_from=src_table,
            old_cluster_table=None,
        )

        cluster_table.where(cluster_table.idx_vector.isin([0, 1, 2])).update(
            {clusterizer.cluster_key: 0}
        )
        cluster_table.where(cluster_table.idx_vector.isin([3, 4, 5])).update(
            {clusterizer.cluster_key: 1}
        )

        all_indices = clusterizer.get_cluster_vector_indices(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
        )
        assert all_indices == {0, 1, 2, 3, 4, 5}

        class0_indices = clusterizer.get_cluster_vector_indices(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            class_key=clusterizer.class_key,
            class_value="0",
        )
        class1_indices = clusterizer.get_cluster_vector_indices(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            class_key=clusterizer.class_key,
            class_value="1",
        )

        assert class0_indices.union(class1_indices) == {0, 1, 2, 3, 4, 5}
        assert class0_indices.isdisjoint(class1_indices)

        with pytest.raises(ValueError, match="must be set together"):
            clusterizer.get_cluster_vector_indices(
                table=cluster_table,
                cluster_key=clusterizer.cluster_key,
                class_key=clusterizer.class_key,
            )
