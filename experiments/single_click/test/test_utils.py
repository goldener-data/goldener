from typing import Any

from unittest.mock import MagicMock

import fiftyone as fo
import PIL
import torch
import numpy as np
from PIL.Image import Image
from omegaconf import OmegaConf
from pathlib import Path

from experiments.single_click import utils


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


class TestPxtSAMSingleClickDatasetImporter:
    def test_importer_next(self, tmp_path: Path) -> None:
        # Setup mocks for pxt_table and exprs
        pxt_table = MagicMock()
        image_expr = MagicMock()
        masks_expr = MagicMock()
        boxes_expr = MagicMock()
        points_expr = MagicMock()
        sam_logits_expr = MagicMock()
        sam_masks_expr = MagicMock()

        # Mock the image expression to have .col.is_stored and .localpath
        image_expr.col.is_stored = True
        image_expr.localpath = str(tmp_path / "image.png")

        # Mock the DataFrame returned by pxt_table.select
        df_mock = MagicMock()

        # Create a dummy image and save it to the local path
        img = PIL.Image.new("RGB", (10, 10))
        img.save(image_expr.localpath)

        # Prepare a fake row: [image, mask, box, point, sam_logits, sam_masks, localpath]
        row = [
            img,
            np.ones((1, 10, 10)),
            np.ones((1, 4)),
            np.ones((1, 1, 2)),
            np.ones((1, 1, 6, 10, 10)),
            np.ones((1, 1, 10, 10)),
            image_expr.localpath,
        ]
        df_mock._output_row_iterator.return_value = iter([row])
        pxt_table.select.return_value = df_mock

        importer = utils.PxtSAMSingleClickDatasetImporter(
            pxt_table=pxt_table,
            image=image_expr,
            masks=masks_expr,
            boxes=boxes_expr,
            points=points_expr,
            sam_logits=sam_logits_expr,
            sam_masks=sam_masks_expr,
            tmp_dir=str(tmp_path),
            dataset_dir=None,
        )

        # Should not raise and should return a tuple
        result = next(importer)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], fo.ImageMetadata)
        assert isinstance(result[2], dict)
        assert len(result[2]) == 10
        assert isinstance(result[2]["ground_truth_0"], fo.Segmentation)
        assert isinstance(result[2]["bounding_box_0"], fo.Detection)
        assert isinstance(result[2]["random_point_0_0"], fo.Keypoint)
        assert isinstance(result[2]["sam_logits_0_0_0"], fo.Heatmap)
        assert isinstance(result[2]["sam_logits_0_0_1"], fo.Heatmap)
        assert isinstance(result[2]["sam_logits_0_0_2"], fo.Heatmap)
        assert isinstance(result[2]["sam_iou_0_0_0"], fo.Classification)
        assert isinstance(result[2]["sam_iou_0_0_1"], fo.Classification)
        assert isinstance(result[2]["sam_iou_0_0_2"], fo.Classification)
        assert isinstance(result[2]["sam_masks_0_0"], fo.Segmentation)
