from typing import Any

import torch
import numpy as np
from PIL.Image import Image
from omegaconf import OmegaConf
from experiments.single_click import utils
from unittest.mock import MagicMock


class TestGetPxtRunName:
    def test_with_run_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"run_name": "run", "dir_name": "dir"},
                "dataset": {"args": {"_target_": "foo.Bar", "split": "train"}},
                "model": {"name": "model"},
            }
        )
        assert utils.get_pxt_run_name(cfg) == "dir.run"

    def test_without_run_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"run_name": None, "dir_name": "dir"},
                "dataset": {
                    "args": {"_target_": "foo.Bar", "split": "train", "category": "cat"}
                },
                "model": {"name": "model"},
            }
        )
        assert utils.get_pxt_run_name(cfg) == "dir.Bar_model_train_cat"

    def test_without_run_name_but_with_category(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"run_name": None, "dir_name": "dir"},
                "dataset": {
                    "args": {"_target_": "foo.Bar", "split": "train", "category": "cat"}
                },
                "model": {"name": "model"},
            }
        )
        assert utils.get_pxt_run_name(cfg) == "dir.Bar_model_train_cat"


class TestGetPxtTableName:
    def test_with_table_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"table_name": "tbl"},
                "pipeline": {"seed": 1, "k_shots": 2, "num_points": 3, "min_area": 4},
            }
        )
        assert utils.get_pxt_table_name(cfg, "run") == "run.tbl"

    def test_without_table_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"table_name": None},
                "pipeline": {"seed": 1, "k_shots": 2, "num_points": 3, "min_area": 4},
            }
        )
        assert utils.get_pxt_table_name(cfg, "run") == "run.seed_1_k_2_n_3_amin_4"


class TestGetPxtTablePath:
    def test_table_path(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {
                    "run_name": "run",
                    "dir_name": "dir",
                    "table_name": "tbl",
                },
                "dataset": {"args": {"_target_": "foo.Bar", "split": "train"}},
                "model": {"name": "model"},
                "pipeline": {"seed": 1, "k_shots": 2, "num_points": 3, "min_area": 4},
            }
        )
        assert utils.get_pxt_table_path(cfg) == "dir.run.tbl"


class TestFormatForPixeltable:
    def test_tensor_image(self) -> None:
        t = torch.ones(3, 8, 8)
        img = utils.format_for_pixeltable(t, is_image=True)
        from PIL.Image import Image

        assert isinstance(img, Image)

    def test_tensor_scalar(self) -> None:
        t = torch.tensor(5)
        val = utils.format_for_pixeltable(t, is_image=False)
        assert val == 5

    def test_tensor_array(self) -> None:
        t = torch.arange(6).reshape(2, 3)
        arr = utils.format_for_pixeltable(t, is_image=False)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3)

    def test_other(self) -> None:
        val = utils.format_for_pixeltable(42, is_image=False)
        assert val == 42


class TestImportAnyDatasetToPixeltable:
    def test_import(self) -> None:
        dataset = MagicMock()
        dataset.__len__.return_value = 2
        dataset.get_raw.side_effect = lambda idx: {
            "image": torch.ones(3, 8, 8),
            "mask": torch.ones(1, 8, 8),
            "label": "foo",
        }

        table = MagicMock()
        table.inserted = []
        table._columns = set()
        table.columns.return_value = table._columns
        table.add_columns.side_effect = lambda d: table._columns.update(d.keys())

        def insert_side_effect(**kwargs: Any) -> None:
            table.inserted.append(kwargs)

        table.insert.side_effect = insert_side_effect

        utils.import_any_dataset_to_pixeltable(
            dataset, table, label="label", num_samples=2, batch_size=1, num_workers=0
        )
        assert len(table.inserted) == 2

        for row in table.inserted:
            assert isinstance(row["image"], Image)
            assert np.all(row["mask"] == np.ones((8, 8)))
            assert row["label"] == "foo"
