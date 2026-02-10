import pytest
import torch

from goldener.clusterize import GoldRandomClusteringTool


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
