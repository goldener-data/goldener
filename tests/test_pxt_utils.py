import pytest
import torch
import time

import pixeltable as pxt

from goldener.pxt_utils import (
    create_pxt_table_from_sample,
    GoldPxtTorchDataset,
    get_array_column_shapes_from_table,
)


@pytest.fixture
def test_table():
    table_path = "test_pxt_utils.test_table"

    # Clean up if exists
    try:
        pxt.drop_dir("test_pxt_utils", force=True)
    except Exception:
        pass

    # Create table
    pxt.create_dir("test_pxt_utils", if_exists="ignore")
    sample = {"data": torch.rand(3, 8, 8), "idx": 0}
    table = create_pxt_table_from_sample(table_path, sample, if_exists="replace_force")

    yield table

    # Cleanup
    try:
        pxt.drop_dir("test_pxt_utils", force=True)
    except Exception:
        pass


class TestGoldPxtTorchDataset:
    def test_cache_cleanup(self, test_table):
        # Create dataset
        dataset = GoldPxtTorchDataset(test_table, keep_cache=False)

        # Store the cache path
        cache_path = dataset.path

        # Verify cache exists
        assert cache_path.exists(), "Cache should exist after dataset creation"
        assert cache_path.is_dir(), "Cache should be a directory"

        # Delete the dataset (triggers __del__)
        del dataset

        # Give time for cleanup
        time.sleep(1)

        # Verify cache was cleaned up
        assert not cache_path.exists(), (
            "Cache should be cleaned up after dataset deletion"
        )

    def test_dataset_iteration_with_shapes(self, test_table):
        shapes = get_array_column_shapes_from_table(test_table)

        dataset = GoldPxtTorchDataset(test_table)

        row_count = 0
        for item in iter(dataset):
            row_count += 1
            assert "data" in item, "Item should contain 'data' key"
            # The shape should match the original shape
            assert item["data"].shape == shapes["data"], (
                "Data should be reshaped correctly"
            )

        assert row_count == test_table.count()

        # Cleanup
        del dataset
        time.sleep(0.1)

    def test_dataset_with_query(self, test_table):
        shapes = get_array_column_shapes_from_table(test_table)

        dataset = GoldPxtTorchDataset(test_table.where(test_table.idx == 0))

        row_count = 0
        for item in iter(dataset):
            row_count += 1
            assert "data" in item, "Item should contain 'data' key"
            # The shape should match the original shape
            assert item["data"].shape == shapes["data"], (
                "Data should be reshaped correctly"
            )

        assert row_count == test_table.count()

        # Cleanup
        del dataset
        time.sleep(0.1)
