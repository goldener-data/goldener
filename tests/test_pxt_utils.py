import pytest
import torch
import time

import pixeltable as pxt

from goldener.pxt_utils import (
    create_pxt_table_from_sample,
    GoldPxtTorchDataset,
    get_array_column_shapes,
)


@pytest.fixture
def test_table():
    """Create a test table with sample data."""
    table_path = "test_pxt_utils.test_table"
    
    # Clean up if exists
    try:
        pxt.drop_dir("test_pxt_utils", force=True)
    except:
        pass
    
    # Create table
    pxt.create_dir("test_pxt_utils", if_exists="ignore")
    sample = {"data": torch.rand(3, 8, 8), "idx": 0}
    table = create_pxt_table_from_sample(table_path, sample, if_exists="replace_force")
    
    yield table
    
    # Cleanup
    try:
        pxt.drop_dir("test_pxt_utils", force=True)
    except:
        pass


class TestGoldPxtTorchDataset:
    def test_cache_cleanup(self, test_table):
        """Test that the cache is cleaned up when GoldPxtTorchDataset is destroyed."""
        # Get array shapes
        shapes = get_array_column_shapes(test_table)
        
        # Create dataset
        dataset = GoldPxtTorchDataset(test_table, shapes)
        
        # Store the cache path
        cache_path = dataset.path
        
        # Verify cache exists
        assert cache_path.exists(), "Cache should exist after dataset creation"
        assert cache_path.is_dir(), "Cache should be a directory"
        
        # Delete the dataset (triggers __del__)
        del dataset
        
        # Give time for cleanup
        time.sleep(0.1)
        
        # Verify cache was cleaned up
        assert not cache_path.exists(), "Cache should be cleaned up after dataset deletion"
    
    def test_dataset_iteration_with_shapes(self, test_table):
        """Test that the dataset properly reshapes arrays during iteration."""
        shapes = get_array_column_shapes(test_table)
        
        # Create dataset
        dataset = GoldPxtTorchDataset(test_table, shapes)
        
        # Iterate through dataset
        items = list(dataset)
        
        # Verify we got data
        assert len(items) > 0, "Dataset should return items"
        
        # Verify data shape is correct
        for item in items:
            assert "data" in item, "Item should contain 'data' key"
            # The shape should match the original shape
            assert item["data"].shape == shapes["data"], "Data should be reshaped correctly"
        
        # Cleanup
        del dataset
        time.sleep(0.1)
