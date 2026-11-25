import re

import numpy as np
import pytest
import torch
import pixeltable as pxt

from sklearn.cluster import BisectingKMeans, Birch, KMeans, MiniBatchKMeans
from torch.utils.data import Dataset

from goldener.clusterize import (
    GoldRandomClusteringTool,
    GoldClusterizer,
    GoldSKLearnClusteringTool,
    get_random_chunk_assignment,
    get_chunk_assignments_from_table,
)
from goldener.reduce import GoldSKLearnReductionTool
from goldener.pxt_utils import GoldPxtTorchDataset
from goldener.torch_utils import collate_keeping_sequences_as_sequences


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


def _same_random_state(left: dict, right: dict) -> bool:
    return bool(
        left["bit_generator"] == right["bit_generator"]
        and np.array_equal(left["state"]["key"], right["state"]["key"])
        and left["state"]["pos"] == right["state"]["pos"]
        and left["has_gauss"] == right["has_gauss"]
        and left["gauss"] == right["gauss"]
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

    def test_constructor_seeds_or_reuses_random_state(self):
        seeded = GoldRandomClusteringTool(42)
        expected = np.random.RandomState(42).get_state(legacy=False)
        assert _same_random_state(seeded.random_state, expected)
        again = seeded.random_state
        assert _same_random_state(seeded.random_state, again)

        caller = np.random.RandomState(3)
        shared = GoldRandomClusteringTool(caller)
        before = shared.random_state
        caller.random()
        assert not _same_random_state(shared.random_state, before)

        fresh = GoldRandomClusteringTool(None)
        first_read = fresh.random_state
        assert _same_random_state(fresh.random_state, first_read)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "The random state must be a NumPy RandomState, an integer or None, "
                "got Generator."
            ),
        ):
            GoldRandomClusteringTool(np.random.default_rng(0))

    @pytest.mark.parametrize(
        ("x", "n_clusters", "message"),
        [
            pytest.param(
                torch.randn(10),
                2,
                "GoldRandomClusteringTool only accepts 2D tensors "
                "(num_vectors, feature_dim). Got shape torch.Size([10]). "
                "Please ensure your input is already vectorized.",
                id="1d",
            ),
            pytest.param(
                torch.randn(10, 5, 3),
                2,
                "GoldRandomClusteringTool only accepts 2D tensors "
                "(num_vectors, feature_dim). Got shape torch.Size([10, 5, 3]). "
                "Please ensure your input is already vectorized.",
                id="3d",
            ),
            pytest.param(
                torch.randn(10, 3),
                15,
                "Number of clusters (15) cannot be greater than "
                "the number of samples (10).",
                id="too_many_clusters",
            ),
        ],
    )
    def test_rejected_fit_does_not_advance_random_state(self, x, n_clusters, message):
        tool = GoldRandomClusteringTool(42)
        state = tool.random_state
        with pytest.raises(ValueError, match=re.escape(message)):
            tool.fit(x, n_clusters)
        assert _same_random_state(tool.random_state, state)

    def test_fit_consumes_random_state_and_restore_replays(self):
        x = torch.randn(10, 3)
        n_clusters = 2
        tool = GoldRandomClusteringTool(42)
        initial = tool.random_state

        first = tool.fit(x, n_clusters)
        advanced = tool.random_state
        assert not _same_random_state(advanced, initial)
        second = tool.fit(x, n_clusters)
        assert not _same_random_state(tool.random_state, advanced)

        replay = GoldRandomClusteringTool(42)
        assert torch.equal(replay.fit(x, n_clusters), first)
        assert torch.equal(replay.fit(x, n_clusters), second)

        returned = tool.random_state
        held = tool.random_state
        returned["state"]["key"][0] = 0
        assert _same_random_state(tool.random_state, held)

        tool.update_random_state(initial)
        assert _same_random_state(tool.random_state, initial)
        assert torch.equal(tool.fit(x, n_clusters), first)
        assert _same_random_state(initial, GoldRandomClusteringTool(42).random_state)

        held_tool = GoldRandomClusteringTool(None)
        before = held_tool.random_state
        held_first = held_tool.fit(x, n_clusters)
        after = held_tool.random_state
        assert not _same_random_state(after, before)
        held_tool.update_random_state(before)
        assert torch.equal(held_tool.fit(x, n_clusters), held_first)
        assert _same_random_state(held_tool.random_state, after)

        single = GoldRandomClusteringTool(42)
        before_single = single.random_state
        single_labels = single.fit(x, 1)
        assert single_labels.dtype == torch.int64
        assert torch.equal(single_labels, torch.zeros(10, dtype=torch.int64))
        assert not _same_random_state(single.random_state, before_single)

        uneven = GoldRandomClusteringTool(42).fit(torch.randn(10, 3), 3)
        assert [uneven.tolist().count(cluster_id) for cluster_id in range(3)] == [
            4,
            3,
            3,
        ]

    @pytest.mark.parametrize(
        ("state", "exc", "message"),
        [
            pytest.param(
                1,
                TypeError,
                "state must be a dict or a tuple.",
                id="not_a_dict",
            ),
            pytest.param(
                {},
                ValueError,
                "state dictionary is not valid.",
                id="empty_dict",
            ),
            pytest.param(
                {"bit_generator": "MT19937"},
                ValueError,
                "state dictionary is not valid.",
                id="incomplete_dict",
            ),
        ],
    )
    def test_rejected_state_does_not_replace_random_state(self, state, exc, message):
        tool = GoldRandomClusteringTool(42)
        before = tool.random_state
        with pytest.raises(exc, match=re.escape(message)):
            tool.update_random_state(state)
        assert _same_random_state(tool.random_state, before)


class TestGetRandomChunkAssignment:
    def test_returns_single_chunk_when_max_chunk_size_is_none(self):
        to_chunk = list(range(10))

        chunks = get_random_chunk_assignment(
            to_chunk, max_chunk_size=None, random_state=0
        )

        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert chunks[0] is to_chunk
        assert chunks[0] == to_chunk

    def test_returns_single_chunk_when_max_chunk_size_ge_length(self):
        to_chunk = list(range(7))

        chunks = get_random_chunk_assignment(
            to_chunk, max_chunk_size=10, random_state=0
        )

        assert len(chunks) == 1
        assert chunks[0] is to_chunk
        assert chunks[0] == to_chunk

    def test_chunks_cover_all_indices_without_duplicates(self):
        to_chunk = list(range(20))
        max_chunk_size = 6

        chunks = get_random_chunk_assignment(
            to_chunk, max_chunk_size=max_chunk_size, random_state=123
        )

        flattened = [idx for ch in chunks for idx in ch]
        assert sorted(flattened) == sorted(to_chunk)
        assert len(flattened) == len(set(flattened))

        for ch in chunks:
            assert 0 < len(ch) <= max_chunk_size

    def test_random_state_makes_assignment_deterministic(self):
        to_chunk = list(range(15))
        max_chunk_size = 4

        chunks1 = get_random_chunk_assignment(
            to_chunk, max_chunk_size=max_chunk_size, random_state=0
        )
        chunks2 = get_random_chunk_assignment(
            to_chunk, max_chunk_size=max_chunk_size, random_state=0
        )

        assert chunks1 == chunks2

    def test_with_0_max(self):
        with pytest.raises(
            ValueError, match="max_chunk_size must be a positive integer"
        ):
            get_random_chunk_assignment(
                list(range(15)), max_chunk_size=0, random_state=0
            )

    def test_with_empty(self):
        with pytest.raises(
            ValueError, match="to_chunk must be a non-empty list of indices"
        ):
            get_random_chunk_assignment([], max_chunk_size=10, random_state=0)


class TestGoldSKLearnClusteringTool:
    def test_init_raises_when_tool_has_no_predict(self):
        class NoPredict:
            def __init__(self) -> None:
                self.n_clusters = 2

        with pytest.raises(ValueError, match="must provide a predict method"):
            GoldSKLearnClusteringTool(NoPredict())

    def test_init_raises_when_tool_has_no_n_clusters(self):
        class NoNClusters:
            def predict(self, x):
                return x

        with pytest.raises(
            NotImplementedError, match="must allow specifying `n_clusters`"
        ):
            GoldSKLearnClusteringTool(NoNClusters())

    def test_fit_returns_tensor_with_expected_shape_and_labels(self):
        from sklearn.cluster import KMeans

        n_samples = 20
        n_clusters = 3
        x = torch.randn(n_samples, 4)

        tool = GoldSKLearnClusteringTool(KMeans(n_clusters=n_clusters, random_state=0))
        labels = tool.fit(x, n_clusters)

        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (n_samples,)
        assert set(labels.tolist()).issubset(set(range(n_clusters)))

    def test_fit_rejects_1d_tensor(self):
        """Test that GoldSKLearnClusteringTool.fit rejects 1D tensor input."""
        from sklearn.cluster import KMeans

        tool = GoldSKLearnClusteringTool(KMeans(n_clusters=2, random_state=0))
        with pytest.raises(
            ValueError, match="GoldSKLearnClusteringTool only accepts 2D tensors"
        ):
            tool.fit(torch.randn(10), n_clusters=2)

    def test_fit_rejects_3d_tensor(self):
        """Test that GoldSKLearnClusteringTool.fit rejects 3D tensor input."""
        from sklearn.cluster import KMeans

        tool = GoldSKLearnClusteringTool(KMeans(n_clusters=2, random_state=0))
        with pytest.raises(
            ValueError, match="GoldSKLearnClusteringTool only accepts 2D tensors"
        ):
            tool.fit(torch.randn(10, 5, 3), n_clusters=2)

    def test_predict_rejects_1d_tensor(self):
        """Test that GoldSKLearnClusteringTool.predict rejects 1D tensor input."""
        from sklearn.cluster import KMeans

        tool = GoldSKLearnClusteringTool(KMeans(n_clusters=2, random_state=0))
        tool.fit(torch.randn(10, 5), n_clusters=2)
        with pytest.raises(
            ValueError, match="GoldSKLearnClusteringTool only accepts 2D tensors"
        ):
            tool.predict(torch.randn(10))

    def test_predict_rejects_3d_tensor(self):
        """Test that GoldSKLearnClusteringTool.predict rejects 3D tensor input."""
        from sklearn.cluster import KMeans

        tool = GoldSKLearnClusteringTool(KMeans(n_clusters=2, random_state=0))
        tool.fit(torch.randn(10, 5), n_clusters=2)
        with pytest.raises(
            ValueError, match="GoldSKLearnClusteringTool only accepts 2D tensors"
        ):
            tool.predict(torch.randn(10, 5, 3))

    def test_predict_matches_sklearn_predict_shape(self):
        from sklearn.cluster import KMeans

        n_samples = 10
        n_clusters = 2
        x = torch.randn(n_samples, 5)

        base = KMeans(n_clusters=n_clusters, random_state=0)
        tool = GoldSKLearnClusteringTool(base)

        tool.fit(x[:5], n_clusters)
        preds = tool.predict(x)

        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (n_samples,)
        assert set(preds.tolist()).issubset(set(range(n_clusters)))

    def test_random_state_initialization(self):
        def assert_seeded_state(state, seed):
            expected = np.random.RandomState(seed).get_state(legacy=False)
            assert state is not None
            assert np.array_equal(state["state"]["key"], expected["state"]["key"])
            assert state["state"]["pos"] == expected["state"]["pos"]

        seeded = KMeans(n_clusters=2, random_state=0)
        tool = GoldSKLearnClusteringTool(seeded)
        assert tool.random_state is None
        assert seeded.random_state == 0

        tool.update_random_state(np.random.RandomState(7).get_state(legacy=False))
        assert_seeded_state(seeded.random_state.get_state(legacy=False), 7)
        assert_seeded_state(tool.random_state, 7)

        caller_state = np.random.RandomState(3)
        reused = KMeans(n_clusters=2, random_state=caller_state)
        assert_seeded_state(GoldSKLearnClusteringTool(reused).random_state, 3)
        assert reused.random_state is caller_state

        override = KMeans(n_clusters=2, random_state=caller_state)
        tool = GoldSKLearnClusteringTool(override, random_state=5)
        assert isinstance(override.random_state, np.random.RandomState)
        assert override.random_state is not caller_state
        assert_seeded_state(tool.random_state, 5)

        explicit = KMeans(n_clusters=2)
        GoldSKLearnClusteringTool(explicit, random_state=caller_state)
        assert explicit.random_state is caller_state

        with pytest.raises(ValueError, match="must be a NumPy RandomState"):
            GoldSKLearnClusteringTool(
                KMeans(n_clusters=2), random_state=np.random.default_rng(0)
            )

        birch = GoldSKLearnClusteringTool(Birch(n_clusters=2))
        assert birch.random_state is None
        with pytest.raises(ValueError, match="has no `random_state` parameter"):
            birch.update_random_state(caller_state.get_state(legacy=False))
        with pytest.raises(ValueError, match="has no `random_state` parameter"):
            GoldSKLearnClusteringTool(Birch(n_clusters=2), random_state=0)

    @pytest.mark.parametrize(
        "make_estimator",
        [
            lambda: KMeans(n_clusters=4),
            lambda: MiniBatchKMeans(n_clusters=4),
            lambda: BisectingKMeans(n_clusters=4),
        ],
        ids=["kmeans", "minibatch_kmeans", "bisecting_kmeans"],
    )
    def test_random_state_roundtrip_restores_assignments(self, make_estimator):
        x = torch.randn(30, 2, generator=torch.Generator().manual_seed(0))
        tool = GoldSKLearnClusteringTool(make_estimator(), random_state=0)
        state = tool.random_state
        assert state is not None

        first = tool.fit(x, 4)
        advanced = tool.random_state
        assert advanced is not None
        assert not np.array_equal(advanced["state"]["key"], state["state"]["key"])

        tool.update_random_state(state)
        assert torch.equal(tool.fit(x, 4), first)

        expected = GoldSKLearnClusteringTool(make_estimator(), random_state=0).fit(x, 4)
        assert torch.equal(first, expected)


class TestGetChunkAssignmentsFromTable:
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def test_saved_chunk_assignments_use_chunk_order_without_repartitioning(self):
        table = pxt.create_table(
            "unit_test.saved_chunks",
            source=[
                {"idx_vector": 40, "chunk": 8, "other": 1},
                {"idx_vector": 10, "chunk": 2, "other": 2},
                {"idx_vector": 70, "chunk": 8, "other": 3},
            ],
        )

        chunks = get_chunk_assignments_from_table(table, max_chunk_size=1)

        assert [set(chunk) for chunk in chunks] == [{10}, {40, 70}]
        assert (
            get_chunk_assignments_from_table(
                table, restrict_to={40, 70}, max_chunk_size=1
            )
            == chunks[1:]
        )

    def test_incomplete_chunk_assignments_are_repartitioned_and_saved(self):
        table = pxt.create_table(
            "unit_test.incomplete_chunks",
            source=[
                {"idx_vector": 40, "chunk": 8},
                {"idx_vector": 10, "chunk": None},
                {"idx_vector": 70, "chunk": 8},
            ],
        )

        chunks = get_chunk_assignments_from_table(
            table, max_chunk_size=2, random_state=0
        )

        assert sorted(idx for chunk in chunks for idx in chunk) == [10, 40, 70]
        assert sorted(map(len, chunks)) == [1, 2]
        assert {row["idx_vector"]: row["chunk"] for row in table.collect()} == {
            idx: chunk_idx for chunk_idx, chunk in enumerate(chunks) for idx in chunk
        }
        assert [set(chunk) for chunk in get_chunk_assignments_from_table(table)] == [
            set(chunk) for chunk in chunks
        ]


class TestGoldClusterizer:
    def setup_method(self):
        pxt.drop_dir("unit_test", force=True)
        pxt.create_dir("unit_test", if_exists="ignore")

    def teardown_method(self):
        pxt.drop_dir("unit_test", force=True)

    def test_distribute_cannot_be_enabled(self):
        error = "Distributed processing is not implemented for GoldClusterizer"
        with pytest.raises(NotImplementedError, match=error):
            GoldClusterizer(
                table_path="unit_test.test_distribute",
                clustering_tool=GoldRandomClusteringTool(),
                distribute=True,
            )

        clusterizer = GoldClusterizer(
            table_path="unit_test.test_distribute",
            clustering_tool=GoldRandomClusteringTool(),
        )
        assert clusterizer.distribute is False

        with pytest.raises(NotImplementedError, match=error):
            clusterizer.distribute = True
        assert clusterizer.distribute is False

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            *[
                pytest.param(
                    {key: "chunk"},
                    f"{key}='chunk' collides with the identity column 'chunk'",
                    id=f"reserved_chunk_{key}",
                )
                for key in ["cluster_key", "vectorized_key", "label_key"]
            ],
            pytest.param(
                {"to_keep_schema": {"chunk": pxt.Int}},
                r"cannot redefine the identity columns \['chunk'\]",
                id="reserved_chunk_kept",
            ),
            pytest.param(
                {"cluster_key": "idx_vector"},
                r"cluster_key='idx_vector' collides with the identity column",
                id="key_is_identity",
            ),
            pytest.param(
                {"label_key": "vectorized"},
                r"share the same column name: 'vectorized'",
                id="duplicate_keys",
            ),
            pytest.param(
                {"to_keep_schema": {"idx_vector": pxt.Int}},
                r"cannot redefine the identity columns \['idx_vector'\]",
                id="kept_identity",
            ),
        ],
    )
    def test_rejects_column_name_collisions(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GoldClusterizer(
                table_path="unit_test.test_collisions",
                clustering_tool=GoldRandomClusteringTool(),
                **kwargs,
            )

    def test_collate_fn_defaults_to_collate_keeping_sequences_as_sequences(self):
        clusterizer = GoldClusterizer(
            table_path="unit_test.cluster_default_collate",
            clustering_tool=GoldRandomClusteringTool(),
        )
        assert clusterizer.collate_fn is collate_keeping_sequences_as_sequences

        def custom_collate_fn(batch):
            return batch

        clusterizer = GoldClusterizer(
            table_path="unit_test.cluster_default_collate",
            clustering_tool=GoldRandomClusteringTool(),
            collate_fn=custom_collate_fn,
        )
        assert clusterizer.collate_fn is custom_collate_fn

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

    @pytest.mark.parametrize("incomplete", [False, True], ids=["reuse", "redraw"])
    @pytest.mark.parametrize("label_key", [None, "label"], ids=["all", "per_label"])
    def test_chunk_assignment_survives_interrupted_clustering(
        self, monkeypatch, incomplete, label_key
    ):
        table_path = "unit_test.interrupted_chunks"
        dataset = DummyDataset(
            [
                {
                    "vectorized": torch.tensor([float(idx), 0.0]),
                    "idx": idx,
                    "label": str(idx % 2),
                }
                for idx in range(24)
            ]
        )
        tool = GoldRandomClusteringTool(random_state=0)
        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=tool,
            chunk=4,
            label_key=label_key,
        )
        original_fit = tool.fit
        fitted_chunks = []

        def interrupted_fit(x, n_clusters):
            if fitted_chunks:
                raise RuntimeError("interrupted clustering")
            fitted_chunks.append(set(x[:, 0].tolist()))
            return original_fit(x, n_clusters)

        monkeypatch.setattr(tool, "fit", interrupted_fit)
        with pytest.raises(RuntimeError, match="interrupted clustering"):
            clusterizer.cluster_in_table(dataset, n_clusters=2)

        table = pxt.get_table(table_path)
        view = pxt.get_table(f"{table_path}_chunks")
        saved = {
            row["idx_vector"]: row["chunk"]
            for row in view.select(view.idx_vector, view.chunk).collect()
        }
        completed = {
            row["idx_vector"]: row["cluster"]
            for row in table.where(table.cluster != None).collect()  # noqa: E711
        }
        assert len(completed) == 4
        pending_query = view.cluster == None  # noqa: E711
        if label_key is not None:
            completed_rows = (
                view.where(view.cluster != None)  # noqa: E711
                .select(view.label)
                .collect()
            )
            pending_query = pending_query & (view.label == completed_rows[0]["label"])
        pending = {row["idx_vector"] for row in view.where(pending_query).collect()}
        assert all(saved[idx] is not None for idx in pending)
        if incomplete:
            view.where(view.idx_vector.isin(pending)).update({"chunk": 99})
            view.where(view.idx_vector == min(pending)).update({"chunk": None})

        draws = []

        def draw_chunks(to_chunk, max_chunk_size, random_state):
            draws.append(set(to_chunk))
            return get_random_chunk_assignment(to_chunk, max_chunk_size, random_state=1)

        def record_fit(x, n_clusters):
            fitted_chunks.append(set(x[:, 0].tolist()))
            return original_fit(x, n_clusters)

        monkeypatch.setattr(
            "goldener.clusterize.get_random_chunk_assignment", draw_chunks
        )
        monkeypatch.setattr(tool, "fit", record_fit)
        clusterizer.cluster_in_table(dataset, n_clusters=2)

        assert table.where(table.cluster == None).count() == 0  # noqa: E711
        assert {
            row["idx_vector"]: row["cluster"]
            for row in table.where(table.idx_vector.isin(completed)).collect()
        } == completed
        assert all(
            row["chunk"] == saved[row["idx_vector"]]
            for row in view.where(view.idx_vector.isin(completed)).collect()
        )
        resumed = {
            row["idx_vector"]: row["chunk"]
            for row in view.where(view.idx_vector.isin(pending)).collect()
        }
        assert (pending in draws) == incomplete
        expected_chunks = {}
        for idx, chunk_idx in resumed.items():
            expected_chunks.setdefault(chunk_idx, set()).add(float(idx))
        for expected in expected_chunks.values():
            assert expected in fitted_chunks[1:]
        if not incomplete:
            assert resumed == {idx: saved[idx] for idx in pending}

    def test_cluster_in_table_force_same_cluster(self):
        src_path = "unit_test.src_cluster_force_table"
        cluster_path = "unit_test.test_cluster_force_same"

        rows = [
            {
                "idx": sample,
                "idx_vector": sample * 3 + vector,
                "vectorized": torch.rand(4).numpy(),
            }
            for sample in range(10)
            for vector in range(3)
        ]
        src_table = pxt.create_table(src_path, source=rows, if_exists="replace_force")

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            chunk=10,
            force_same_cluster=True,
        )

        cluster_table = clusterizer.cluster_in_table(src_table, n_clusters=4)

        assert cluster_table.count() == 30

        samples_by_cluster = [
            clusterizer.get_cluster_indices(
                cluster_table,
                cluster_key=clusterizer.cluster_key,
                cluster_idx=cluster_idx,
                idx_key="idx",
            )
            for cluster_idx in range(4)
        ]
        seen_samples: set[int] = set()
        for samples in samples_by_cluster:
            assert seen_samples.isdisjoint(samples)
            seen_samples.update(samples)
        assert seen_samples == set(range(10))

    def test_cluster_in_table_without_force_keeps_per_vector_labels(self):
        src_path = "unit_test.src_cluster_no_force_table"
        cluster_path = "unit_test.test_cluster_no_force"

        rows = [
            {
                "idx": sample,
                "idx_vector": sample * 3 + vector,
                "vectorized": torch.rand(4).numpy(),
            }
            for sample in range(10)
            for vector in range(3)
        ]
        src_table = pxt.create_table(src_path, source=rows, if_exists="replace_force")

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
        )

        cluster_table = clusterizer.cluster_in_table(src_table, n_clusters=4)

        assert cluster_table.count() == 30
        labeled_vectors = (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx_vector)
            .distinct()
            .count()
        )
        assert labeled_vectors == 30

    def test_cluster_from_table_with_restrict_to(self):
        src_path = "unit_test.src_cluster_restrict_table"
        cluster_path = "unit_test.test_cluster_restrict"

        src_table = self._make_src_table(src_path, n=10)

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
        )

        cluster_table = clusterizer.cluster_in_table(
            src_table, n_clusters=3, restrict_to={2, 5, 7}
        )

        assert cluster_table.count() == 3
        clustered = (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .collect()
        )
        assert {row["idx"] for row in clustered} == {2, 5, 7}

    def test_cluster_from_table_with_restrict_to_custom_index_key(self):
        src_path = "unit_test.src_cluster_restrict_custom_idx_table"
        cluster_path = "unit_test.test_cluster_restrict_custom_idx"

        source_rows = [
            {
                "idx": i,
                "idx_vector": i * 10,
                "vectorized": torch.rand(4).numpy(),
            }
            for i in range(10)
        ]
        src_table = pxt.create_table(
            src_path, source=source_rows, if_exists="replace_force"
        )

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
        )

        cluster_table = clusterizer.cluster_in_table(
            src_table,
            n_clusters=3,
            restrict_to={2, 5, 7},
            restriction_idx_key="idx",
        )

        assert cluster_table.count() == 3
        clustered = (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .collect()
        )
        assert {row["idx"] for row in clustered} == {2, 5, 7}

    def test_cluster_from_dataset_with_restrict_to(self):
        cluster_path = "unit_test.test_cluster_restrict_dataset"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(10)]
        )

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
        )

        clusterizer.cluster_in_dataset(dataset, n_clusters=3, restrict_to={2, 5, 7})

        cluster_table = pxt.get_table(cluster_path)
        assert cluster_table.count() == 3
        clustered = (
            cluster_table.where(
                cluster_table[clusterizer.cluster_key] != None  # noqa: E711
            )
            .select(cluster_table.idx)
            .distinct()
            .collect()
        )
        assert {row["idx"] for row in clustered} == {2, 5, 7}

    def test_cluster_in_table_with_invalid_restriction_idx_key(self):
        table_path = "unit_test.test_cluster_invalid_restriction_idx_key"
        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
        )

        with pytest.raises(
            ValueError,
            match="restriction_idx_key must be either 'idx' or 'idx_vector'",
        ):
            clusterizer.cluster_in_table(
                DummyDataset([]),
                n_clusters=2,
                restrict_to={0},
                restriction_idx_key="invalid",
            )

        with pytest.raises(pxt.Error):
            pxt.get_table(table_path)

    def test_cluster_in_table_with_valid_restriction_idx_key(self):
        table_path = "unit_test.test_cluster_restrict_idx"
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(10)]
        )
        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            batch_size=5,
        )

        cluster_table = clusterizer.cluster_in_table(
            dataset,
            n_clusters=3,
            restrict_to={2, 5, 7},
            restriction_idx_key="idx",
        )

        assert cluster_table.count() == 3
        assert {
            row["idx"] for row in cluster_table.select(cluster_table.idx).collect()
        } == {2, 5, 7}

    def test_cluster_in_table_with_reducer(self):
        table_path = "unit_test.test_cluster_reducer"

        dataset = DummyDataset(
            [{"vectorized": torch.rand(8), "idx": idx} for idx in range(30)]
        )

        from sklearn.decomposition import PCA

        reducer = GoldSKLearnReductionTool(PCA(n_components=4))

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
        with pytest.raises(pxt.Error):
            pxt.get_table(f"{table_path}_chunks")

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

    @pytest.mark.parametrize(
        "chunk, expected_chunk_count",
        [
            pytest.param(None, 1, id="unchunked"),
            pytest.param(10, 3, id="multiple_chunks"),
            pytest.param(100, 1, id="chunk_larger_than_population"),
        ],
    )
    def test_cluster_in_table_with_restart_allowed(self, chunk, expected_chunk_count):
        table_path = "unit_test.test_cluster_restart_allowed"
        dataset = DummyDataset(
            [{"vectorized": torch.rand(4), "idx": idx} for idx in range(30)]
        )
        clusterizer = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=False,
            batch_size=5,
            chunk=chunk,
        )

        cluster_table = clusterizer.cluster_in_table(dataset, n_clusters=3)
        view = pxt.get_table(f"{table_path}_chunks")
        assert set(view.columns()) == set(cluster_table.columns()) | {"chunk"}
        assert "chunk" not in cluster_table.columns()
        saved = {
            row["idx_vector"]: (row["idx"], row["cluster"], row["chunk"])
            for row in view.select(
                view.idx_vector, view.idx, view.cluster, view.chunk
            ).collect()
        }
        assert cluster_table.count() == len(saved) == 30
        assert {idx for idx, _, _ in saved.values()} == set(range(30))
        assert {cluster for _, cluster, _ in saved.values()} <= set(range(3))
        chunk_sizes = {}
        for _, _, chunk_idx in saved.values():
            assert isinstance(chunk_idx, int)
            chunk_sizes[chunk_idx] = chunk_sizes.get(chunk_idx, 0) + 1
        assert len(chunk_sizes) == expected_chunk_count
        assert max(chunk_sizes.values()) <= (chunk if chunk is not None else 30)

        clusterizer_restart = GoldClusterizer(
            table_path=table_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            batch_size=5,
            chunk=chunk,
        )
        restarted_table = clusterizer_restart.cluster_in_table(dataset, n_clusters=3)

        assert restarted_table.count() == 30
        assert {
            row["idx_vector"]: (row["idx"], row["cluster"], row["chunk"])
            for row in view.select(
                view.idx_vector, view.idx, view.cluster, view.chunk
            ).collect()
        } == saved

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
            label_key="label",
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

        label_col = cluster_table[clusterizer.label_key]
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

        for cluster_idx in range(4):
            assert (
                clusterizer.get_cluster_count(
                    table=cluster_table,
                    cluster_key=clusterizer.cluster_key,
                    cluster_idx=cluster_idx,
                )
                > 0
            )

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

    def test_get_cluster_indices(self):
        src_path = "unit_test.test_cluster_indices_explicit_src"
        cluster_path = "unit_test.test_cluster_indices_explicit"

        src_table = self._make_src_table(src_path, n=6, with_class=True)

        clusterizer = GoldClusterizer(
            table_path=cluster_path,
            clustering_tool=GoldRandomClusteringTool(random_state=0),
            allow_existing=True,
            label_key="label",
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

        assert (
            clusterizer.get_cluster_count(
                table=cluster_table,
                cluster_key=clusterizer.cluster_key,
            )
            == 0
        )

        class0_indices = clusterizer.get_cluster_indices(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            cluster_idx=0,
            label_key=clusterizer.label_key,
            label_value="0",
        )
        assert class0_indices == {
            0,
            2,
        }
        assert len(class0_indices) == clusterizer.get_cluster_count(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            cluster_idx=0,
            label_key=clusterizer.label_key,
            label_value="0",
        )

        class1_indices = clusterizer.get_cluster_indices(
            table=cluster_table,
            cluster_idx=0,
            cluster_key=clusterizer.cluster_key,
            label_key=clusterizer.label_key,
            label_value="1",
        )
        assert len(class1_indices) == clusterizer.get_cluster_count(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            cluster_idx=0,
            label_key=clusterizer.label_key,
            label_value="1",
        )

        assert class1_indices == {
            1,
        }

        cluster0_indices = clusterizer.get_cluster_indices(
            table=cluster_table,
            cluster_idx=0,
            cluster_key=clusterizer.cluster_key,
        )
        assert len(cluster0_indices) == clusterizer.get_cluster_count(
            table=cluster_table,
            cluster_key=clusterizer.cluster_key,
            cluster_idx=0,
        )

        with pytest.raises(ValueError, match="must be set together"):
            clusterizer.get_cluster_indices(
                table=cluster_table,
                cluster_key=clusterizer.cluster_key,
                label_key=clusterizer.label_key,
            )
