import gc

import pytest
import torch
import numpy as np

import pixeltable as pxt

from goldener.pxt_utils import (
    create_pxt_table_from_sample,
    GoldPxtTorchDataset,
    get_expr_from_column_name,
    create_pxt_dirs_for_path,
    set_value_to_idx_rows,
    pxt_torch_dataset_collate_fn,
    get_distinct_value_and_count_in_column,
    get_column_distinct_ratios,
    get_array_column_shapes_from_table,
    get_sample_row_from_idx,
    get_valid_table,
    make_batch_ready_for_table,
    get_pxt_table_primary_keys,
    check_pxt_table_has_primary_key,
    get_max_value_in_column,
)


@pytest.fixture
def test_table():
    table_path = "test_pxt_utils.test_table"

    pxt.create_dir("test_pxt_utils", if_exists="ignore")
    sample = [{"data": np.random.rand(3, 8, 8).astype(np.float32), "idx": 0}]
    table = pxt.create_table(table_path, source=sample, if_exists="replace_force")

    yield table

    pxt.drop_dir("test_pxt_utils", force=True)


class TestCreatePxtTableFromSample:
    def test_create_table_from_sample_with_torch_tensor(self):
        table_path = "test_create_from_sample.test_table"

        sample = {"data": torch.ones(3, 8, 8), "idx": 0}
        table = create_pxt_table_from_sample(
            table_path, sample, if_exists="replace_force"
        )

        assert table is not None
        assert table.columns() == ["data", "idx"]

        pxt.drop_dir("test_create_from_sample", force=True)

    def test_create_table_from_sample_with_numpy_array(self):
        table_path = "test_create_from_sample_np.test_table"

        sample = {"data": np.zeros((3, 8, 8), dtype=np.float32), "idx": 0}
        table = create_pxt_table_from_sample(
            table_path, sample, if_exists="replace_force"
        )

        assert table is not None
        assert table.columns() == ["data", "idx"]

        pxt.drop_dir("test_create_from_sample_np", force=True)

    def test_create_table_from_sample_with_add(self):
        table_path = "test_create_from_sample_np.test_table"

        sample = {"data": np.zeros((3, 8, 8), dtype=np.float32), "idx": 0}
        table = create_pxt_table_from_sample(
            table_path, sample, add={"new": 2}, if_exists="replace_force"
        )

        assert table is not None
        assert table.columns() == ["data", "idx", "new"]

        pxt.drop_dir("test_create_from_sample_np", force=True)

    def test_create_table_from_sample_with_unwrap(self):
        table_path = "test_create_from_sample_np.test_table"

        sample = {"data": np.zeros((3, 8, 8), dtype=np.float32), "idx": [0]}
        table = create_pxt_table_from_sample(
            table_path, sample, unwrap=True, if_exists="replace_force"
        )

        assert table is not None
        assert table.columns() == ["data", "idx"]

        pxt.drop_dir("test_create_from_sample_np", force=True)


class TestGoldPxtTorchDataset:
    def test_cache_cleanup(self, test_table):
        dataset = GoldPxtTorchDataset(test_table, keep_cache=False)
        cache_path = dataset.path

        assert cache_path.exists(), "Cache should exist after dataset creation"
        assert cache_path.is_dir(), "Cache should be a directory"

        del dataset
        gc.collect()

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
            assert item["data"].shape == shapes["data"], (
                "Data should be reshaped correctly"
            )

        assert row_count == test_table.count()

    def test_dataset_with_query(self, test_table):
        shapes = get_array_column_shapes_from_table(test_table)

        dataset = GoldPxtTorchDataset(test_table.where(test_table.idx == 0))

        row_count = 0
        for item in iter(dataset):
            row_count += 1
            assert "data" in item, "Item should contain 'data' key"
            assert item["data"].shape == shapes["data"], (
                "Data should be reshaped correctly"
            )

        assert row_count == test_table.count()


class TestGetExprFromColumnName:
    def test_valid_column_name(self, test_table):
        col_expr = get_expr_from_column_name(test_table, "idx")
        assert col_expr is not None
        assert col_expr.display_str() == "idx"

    def test_invalid_column_name(self, test_table):
        with pytest.raises(ValueError, match="Column 'invalid_column' does not exist"):
            get_expr_from_column_name(test_table, "invalid_column")


class TestCreatePxtDirsForPath:
    def test_multi_level_path(self):
        table_path = "test_dir1.test_dir2.test_table"
        pxt.drop_dir("test_dir1", force=True)

        create_pxt_dirs_for_path(table_path)

        dirs = pxt.list_dirs()
        assert "test_dir1" in dirs

        dirs = pxt.list_dirs("test_dir1")
        assert "test_dir1.test_dir2" in dirs

        pxt.drop_dir("test_dir1", force=True)


class TestSetValueToIdxRows:
    def test_set_value_to_multiple_rows(self):
        pxt.drop_dir("test_set_value", force=True)
        table_path = "test_set_value.test_table"

        pxt.create_dir("test_set_value", if_exists="ignore")
        samples = [
            {"idx": 0, "value": 1},
            {"idx": 1, "value": 2},
            {"idx": 2, "value": 3},
        ]
        table = pxt.create_table(table_path, source=samples)

        table.add_column(label=pxt.String)
        table.update({"label": "A"})

        set_value_to_idx_rows(
            table=table,
            col_expr=table.label,
            idx_expr=table.idx,
            indices={0, 2},
            value="B",
        )

        result_b = table.where(table.label == "B").select(table.idx).collect()
        idx_values = sorted([row["idx"] for row in result_b])
        assert idx_values == [0, 2]

        result_a = table.where(table.label == "A").select(table.idx).collect()
        assert len(result_a) == 1
        assert result_a[0]["idx"] == 1

        pxt.drop_dir("test_set_value", force=True)


class TestPxtTorchDatasetCollateFn:
    def test_collate_numpy_arrays(self):
        batch = [
            {"image": np.array([1, 2, 3]), "label": 0, "text": "hello"},
            {"image": np.array([4, 5, 6]), "label": 1, "text": "world"},
        ]

        result = pxt_torch_dataset_collate_fn(batch)

        assert "image" in result
        assert "label" in result
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["label"], torch.Tensor)
        assert isinstance(result["text"], list)
        assert result["text"] == ["hello", "world"]
        assert result["image"].shape == (2, 3)
        assert torch.equal(result["label"], torch.tensor([0, 1], dtype=torch.int64))

    def test_collate_empty_batch(self):
        batch = []
        result = pxt_torch_dataset_collate_fn(batch)
        assert result == {}


class TestGetDistinctValueAndCountInColumn:
    def test_get_distinct_values(self):
        table_path = "test_distinct.test_table"

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
        pxt.drop_dir("test_distinct", force=True)


class TestGetColumnDistinctRatios:
    def test_get_ratios(self):
        table_path = "test_ratios.test_table"
        pxt.create_dir("test_ratios", if_exists="ignore")

        samples = [
            {"idx": 0, "category": "A"},
            {"idx": 1, "category": "A"},
            {"idx": 2, "category": "B"},
            {"idx": 3, "category": "B"},
        ]
        table = pxt.create_table(table_path, source=samples)

        col_expr = get_expr_from_column_name(table, "category")
        result = get_column_distinct_ratios(table, col_expr)

        assert "A" in result
        assert "B" in result
        assert result["A"] == 0.5
        assert result["B"] == 0.5

        pxt.drop_dir("test_ratios", force=True)


class TestGetSampleRowFromIdx:
    def test_get_sample_row(self, test_table):
        row = get_sample_row_from_idx(test_table, 0, expected_keys=["data"])
        assert row is not None
        assert "idx" in row
        assert row["idx"] == 0

    def test_no_sample(self):
        pxt.create_dir("test_sample_row", if_exists="ignore")
        table = pxt.create_table("test_sample_row.no_sample", source=[{"idx": 1}])

        with pytest.raises(ValueError, match="No sample found at the index 0."):
            get_sample_row_from_idx(table, 0)

        pxt.drop_dir("test_sample_row", force=True)

    def test_more_than_1_sample(self):
        pxt.create_dir("test_sample_row", if_exists="ignore")
        table = pxt.create_table(
            "test_sample_row.more_than_one", source=[{"idx": 0}, {"idx": 0}]
        )

        with pytest.raises(
            ValueError, match="Multiple samples found at the specified index."
        ):
            get_sample_row_from_idx(table, 0)

        pxt.drop_dir("test_sample_row", force=True)


class TestGetValidTable:
    def test_with_existing(self):
        pxt.create_dir("test_get_valid_view", if_exists="ignore")
        samples = [{"idx": 0, "a": 1, "b": 2}]
        table = pxt.create_table("test_get_valid_view.base_table", source=samples)

        out = get_valid_table(table, minimal_schema={"idx": pxt.Int})
        assert out is table

        pxt.drop_dir("test_get_valid_view", force=True)

    def test_existing_with_missing(self):
        pxt.create_dir("test_get_valid_view", if_exists="ignore")
        samples = [{"idx": 0, "a": 1, "b": 2}]
        table = pxt.create_table("test_get_valid_view.base_table", source=samples)

        with pytest.raises(ValueError, match="The table is missing required"):
            get_valid_table(table, minimal_schema={"not present": pxt.Int})

        pxt.drop_dir("test_get_valid_view", force=True)

    def test_with_path(self):
        table = get_valid_table(
            "test_get_valid_view.base_table", minimal_schema={"idx": pxt.Int}
        )

        assert table.columns() == ["idx"]
        pxt.drop_dir("test_get_valid_view", force=True)


class TestMakeBatchReadyForTable:
    def test_with_index_not_present(self):
        batch = {
            "data": np.random.rand(2, 8, 8).astype(np.float32),
            "idx": [0, 1],
            "extra": [10, 20],
        }

        batch_list = make_batch_ready_for_table(
            batch,
            to_insert=["data"],
            index_key="idx",
        )

        assert len(batch_list) == 2
        assert "data" in batch_list[0]
        assert "idx" in batch_list[0]
        assert "extra" not in batch_list[0]

    def test_with_no_index(self):
        batch = {"data": np.random.rand(2, 8, 8).astype(np.float32), "extra": [10, 20]}
        with pytest.raises(
            KeyError,
        ):
            make_batch_ready_for_table(
                batch,
                to_insert=["data"],
                index_key="idx",
            )


class TestGetPxtTablePrimaryKeys:
    def test_with_primary_key(self):
        pxt.drop_dir("unit_test", force=True)

        pxt.create_dir("unit_test", if_exists="ignore")
        table = pxt.create_table(
            "unit_test.table_with_pk",
            schema={"idx": pxt.Required[pxt.Int], "label": pxt.String},
            primary_key="idx",
        )

        pk = get_pxt_table_primary_keys(table)
        assert pk == set(["idx"])

        pxt.drop_dir("unit_test", force=True)

    def test_without_primary_key(self):
        pxt.drop_dir("unit_test", force=True)

        pxt.create_dir("unit_test", if_exists="ignore")
        table = pxt.create_table(
            "unit_test.table_with_pk",
            schema={"idx": pxt.Required[pxt.Int], "label": pxt.String},
        )

        pk = get_pxt_table_primary_keys(table)
        assert pk == set()

        pxt.drop_dir("unit_test", force=True)


class TestCheckPxtTableHasPrimaryKey:
    def test_with_primary_key(self):
        pxt.drop_dir("unit_test", force=True)

        pxt.create_dir("unit_test", if_exists="ignore")
        table = pxt.create_table(
            "unit_test.table_with_pk",
            schema={"idx": pxt.Required[pxt.Int], "label": pxt.String},
            primary_key="idx",
        )
        check_pxt_table_has_primary_key(table, set(["idx"]))

        pxt.drop_dir("unit_test", force=True)

    def test_without_primary_key(self):
        pxt.drop_dir("unit_test", force=True)

        pxt.create_dir("unit_test", if_exists="ignore")
        table = pxt.create_table(
            "unit_test.table_with_pk",
            schema={"idx": pxt.Required[pxt.Int], "label": pxt.String},
        )

        with pytest.raises(
            ValueError, match="does not have all specified columns as primary"
        ):
            check_pxt_table_has_primary_key(table, set(["idx"]))

        pxt.drop_dir("unit_test", force=True)


class TestGetMaxValueInColumn:
    def test_get_max_value(self):
        pxt.drop_dir("test_max_value", force=True)

        table_path = "test_max_value.test_table"

        pxt.create_dir("test_max_value", if_exists="ignore")
        samples = [
            {"idx": 0, "value": 10},
            {"idx": 1, "value": 20},
            {"idx": 2, "value": 15},
        ]
        table = pxt.create_table(table_path, source=samples)

        col_expr = get_expr_from_column_name(table, "value")
        max_value = get_max_value_in_column(table, col_expr)

        assert max_value == 20
        pxt.drop_dir("test_max_value", force=True)

    def test_with_not_int(self):
        pxt.drop_dir("test_max_value", force=True)

        table_path = "test_max_value.test_table"

        pxt.create_dir("test_max_value", if_exists="ignore")
        samples = [
            {"idx": 0, "value": 10.0},
            {"idx": 1, "value": 20.0},
            {"idx": 2, "value": 15.0},
        ]
        table = pxt.create_table(table_path, source=samples)

        col_expr = get_expr_from_column_name(table, "value")

        with pytest.raises(TypeError, match="The maximum value is not an integer"):
            get_max_value_in_column(table, col_expr)

        pxt.drop_dir("test_max_value", force=True)
