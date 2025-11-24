import pytest
import torch
import time
import numpy as np

import pixeltable as pxt

from goldener.pxt_utils import (
    create_pxt_table_from_sample,
    GoldPxtTorchDataset,
    get_array_column_shapes,
    get_expr_from_column_name,
    create_pxt_dirs_for_path,
    set_value_to_idx_rows,
    update_column_if_too_many,
    pxt_torch_dataset_collate_fn,
    get_distinct_value_and_count_in_column,
    _get_column_distinct_ratios,
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
        assert not cache_path.exists(), (
            "Cache should be cleaned up after dataset deletion"
        )

    def test_dataset_iteration_with_shapes(self, test_table):
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
            assert item["data"].shape == shapes["data"], (
                "Data should be reshaped correctly"
            )

        # Cleanup
        del dataset
        time.sleep(0.1)


class TestGetExprFromColumnName:
    def test_valid_column_name(self, test_table):
        col_expr = get_expr_from_column_name(test_table, "idx")
        assert col_expr is not None
        assert col_expr.display_str() == "idx"

    def test_invalid_column_name(self, test_table):
        with pytest.raises(ValueError, match="Column 'invalid_column' does not exist"):
            get_expr_from_column_name(test_table, "invalid_column")

    def test_data_column_name(self, test_table):
        col_expr = get_expr_from_column_name(test_table, "data")
        assert col_expr is not None
        assert col_expr.display_str() == "data"


class TestCreatePxtDirsForPath:
    def test_single_level_path(self):
        table_path = "test_create_dirs.table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_create_dirs", force=True)
        except Exception:
            pass
        
        create_pxt_dirs_for_path(table_path)
        
        # Verify directory was created
        dirs = pxt.list_dirs()
        assert "test_create_dirs" in dirs
        
        # Cleanup
        pxt.drop_dir("test_create_dirs", force=True)

    def test_multi_level_path(self):
        table_path = "test_dir1.test_dir2.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_dir1", force=True)
        except Exception:
            pass
        
        create_pxt_dirs_for_path(table_path)
        
        # Verify directories were created
        dirs = pxt.list_dirs()
        assert "test_dir1" in dirs
        
        # Cleanup
        pxt.drop_dir("test_dir1", force=True)

    def test_already_exists(self):
        table_path = "test_existing.table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_existing", force=True)
        except Exception:
            pass
        
        # Create once
        create_pxt_dirs_for_path(table_path)
        
        # Create again - should not raise
        create_pxt_dirs_for_path(table_path)
        
        # Cleanup
        pxt.drop_dir("test_existing", force=True)


class TestSetValueToIdxRows:
    def test_set_value_to_single_row(self, test_table):
        # Add a column to update
        test_table.add_column(status=pxt.String, stored=True)
        test_table.update({"status": "initial"})
        
        # Get column expression
        col_expr = get_expr_from_column_name(test_table, "status")
        
        # Set value for idx 0
        set_value_to_idx_rows(test_table, col_expr, {0}, "updated")
        
        # Verify the change
        result = test_table.where(test_table.idx == 0).select(test_table.status).collect()
        assert len(result) == 1
        assert result[0]["status"] == "updated"

    def test_set_value_to_multiple_rows(self):
        table_path = "test_set_value.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_set_value", force=True)
        except Exception:
            pass
        
        # Create table with multiple rows
        pxt.create_dir("test_set_value", if_exists="ignore")
        samples = [
            {"idx": 0, "value": 1},
            {"idx": 1, "value": 2},
            {"idx": 2, "value": 3},
        ]
        table = pxt.create_table(table_path, source=samples)
        
        # Add a column to update
        table.add_column(label=pxt.String, stored=True)
        table.update({"label": "A"})
        
        # Get column expression
        col_expr = get_expr_from_column_name(table, "label")
        
        # Set value for idx 0 and 2
        set_value_to_idx_rows(table, col_expr, {0, 2}, "B")
        
        # Verify the changes
        result_b = table.where(table.label == "B").select(table.idx).collect()
        idx_values = sorted([row["idx"] for row in result_b])
        assert idx_values == [0, 2]
        
        result_a = table.where(table.label == "A").select(table.idx).collect()
        assert len(result_a) == 1
        assert result_a[0]["idx"] == 1
        
        # Cleanup
        pxt.drop_dir("test_set_value", force=True)


class TestUpdateColumnIfTooMany:
    def test_no_update_when_under_max(self):
        table_path = "test_update_column.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_update_column", force=True)
        except Exception:
            pass
        
        # Create table with samples
        pxt.create_dir("test_update_column", if_exists="ignore")
        samples = [
            {"idx": 0, "category": "A"},
            {"idx": 1, "category": "A"},
            {"idx": 2, "category": "B"},
        ]
        table = pxt.create_table(table_path, source=samples)
        
        col_expr = get_expr_from_column_name(table, "category")
        
        # Max count is 3, current count is 2
        update_column_if_too_many(table, col_expr, "A", 3, "C")
        
        # Verify no change
        result = table.where(table.category == "A").count()
        assert result == 2
        
        # Cleanup
        pxt.drop_dir("test_update_column", force=True)

    def test_update_when_over_max(self):
        table_path = "test_update_over.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_update_over", force=True)
        except Exception:
            pass
        
        # Create table with samples
        pxt.create_dir("test_update_over", if_exists="ignore")
        samples = [
            {"idx": 0, "category": "A"},
            {"idx": 1, "category": "A"},
            {"idx": 2, "category": "A"},
            {"idx": 3, "category": "A"},
            {"idx": 4, "category": "B"},
        ]
        table = pxt.create_table(table_path, source=samples)
        
        col_expr = get_expr_from_column_name(table, "category")
        
        # Max count is 2, current count is 4
        update_column_if_too_many(table, col_expr, "A", 2, "C")
        
        # Verify changes: should have 2 A's and 2 C's
        count_a = table.where(table.category == "A").count()
        count_c = table.where(table.category == "C").count()
        assert count_a == 2
        assert count_c == 2
        
        # Cleanup
        pxt.drop_dir("test_update_over", force=True)


class TestPxtTorchDatasetCollateFn:
    def test_collate_numpy_arrays(self):
        batch = [
            {"image": np.array([1, 2, 3]), "label": 0},
            {"image": np.array([4, 5, 6]), "label": 1},
        ]
        
        result = pxt_torch_dataset_collate_fn(batch)
        
        assert "image" in result
        assert "label" in result
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["label"], torch.Tensor)
        assert result["image"].shape == (2, 3)
        assert torch.equal(result["label"], torch.tensor([0, 1], dtype=torch.int64))

    def test_collate_integers(self):
        batch = [
            {"idx": 0, "count": 5},
            {"idx": 1, "count": 10},
        ]
        
        result = pxt_torch_dataset_collate_fn(batch)
        
        assert isinstance(result["idx"], torch.Tensor)
        assert isinstance(result["count"], torch.Tensor)
        assert torch.equal(result["idx"], torch.tensor([0, 1], dtype=torch.int64))
        assert torch.equal(result["count"], torch.tensor([5, 10], dtype=torch.int64))

    def test_collate_strings_as_list(self):
        batch = [
            {"text": "hello", "label": 0},
            {"text": "world", "label": 1},
        ]
        
        result = pxt_torch_dataset_collate_fn(batch)
        
        assert "text" in result
        assert isinstance(result["text"], list)
        assert result["text"] == ["hello", "world"]
        assert isinstance(result["label"], torch.Tensor)

    def test_collate_mixed_types(self):
        batch = [
            {"array": np.array([1.0, 2.0]), "num": 5, "text": "a"},
            {"array": np.array([3.0, 4.0]), "num": 10, "text": "b"},
        ]
        
        result = pxt_torch_dataset_collate_fn(batch)
        
        assert isinstance(result["array"], torch.Tensor)
        assert isinstance(result["num"], torch.Tensor)
        assert isinstance(result["text"], list)
        assert result["text"] == ["a", "b"]

    def test_collate_empty_batch(self):
        batch = []
        
        result = pxt_torch_dataset_collate_fn(batch)
        
        assert result == {}


class TestGetDistinctValueAndCountInColumn:
    def test_get_distinct_values(self):
        table_path = "test_distinct.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_distinct", force=True)
        except Exception:
            pass
        
        # Create table with samples
        pxt.create_dir("test_distinct", if_exists="ignore")
        samples = [
            {"idx": 0, "category": "A"},
            {"idx": 1, "category": "A"},
            {"idx": 2, "category": "B"},
            {"idx": 3, "category": "C"},
            {"idx": 4, "category": "C"},
            {"idx": 5, "category": "C"},
        ]
        table = pxt.create_table(table_path, source=samples)
        
        col_expr = get_expr_from_column_name(table, "category")
        result = get_distinct_value_and_count_in_column(table, col_expr)
        
        assert result == {"A": 2, "B": 1, "C": 3}
        
        # Cleanup
        pxt.drop_dir("test_distinct", force=True)

    def test_single_value(self):
        table_path = "test_single_value.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_single_value", force=True)
        except Exception:
            pass
        
        # Create table with same value
        pxt.create_dir("test_single_value", if_exists="ignore")
        samples = [
            {"idx": 0, "label": "X"},
            {"idx": 1, "label": "X"},
        ]
        table = pxt.create_table(table_path, source=samples)
        
        col_expr = get_expr_from_column_name(table, "label")
        result = get_distinct_value_and_count_in_column(table, col_expr)
        
        assert result == {"X": 2}
        
        # Cleanup
        pxt.drop_dir("test_single_value", force=True)


class TestGetColumnDistinctRatios:
    def test_get_ratios(self):
        table_path = "test_ratios.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_ratios", force=True)
        except Exception:
            pass
        
        # Create table with samples
        pxt.create_dir("test_ratios", if_exists="ignore")
        samples = [
            {"idx": 0, "category": "A"},
            {"idx": 1, "category": "A"},
            {"idx": 2, "category": "B"},
            {"idx": 3, "category": "B"},
        ]
        table = pxt.create_table(table_path, source=samples)
        
        col_expr = get_expr_from_column_name(table, "category")
        result = _get_column_distinct_ratios(table, col_expr)
        
        assert "A" in result
        assert "B" in result
        assert abs(result["A"] - 0.5) < 0.01
        assert abs(result["B"] - 0.5) < 0.01
        
        # Cleanup
        pxt.drop_dir("test_ratios", force=True)

    def test_unequal_ratios(self):
        table_path = "test_unequal_ratios.test_table"
        
        # Clean up if exists
        try:
            pxt.drop_dir("test_unequal_ratios", force=True)
        except Exception:
            pass
        
        # Create table with unequal distribution
        pxt.create_dir("test_unequal_ratios", if_exists="ignore")
        samples = [
            {"idx": 0, "label": "X"},
            {"idx": 1, "label": "X"},
            {"idx": 2, "label": "X"},
            {"idx": 3, "label": "Y"},
        ]
        table = pxt.create_table(table_path, source=samples)
        
        col_expr = get_expr_from_column_name(table, "label")
        result = _get_column_distinct_ratios(table, col_expr)
        
        assert "X" in result
        assert "Y" in result
        assert abs(result["X"] - 0.75) < 0.01
        assert abs(result["Y"] - 0.25) < 0.01
        
        # Cleanup
        pxt.drop_dir("test_unequal_ratios", force=True)
