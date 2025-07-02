import os
import random
from typing import Any, Optional

import cv2
import fiftyone as fo
import pixeltable as pxt
import puremagic
from PIL.Image import Image
from bson import ObjectId
from fiftyone.utils import data as foud

from omegaconf import DictConfig

from torchvision.transforms.v2 import functional as F
import numpy as np
import torch
from pixeltable.type_system import ColumnType
from torch.utils.data import DataLoader, RandomSampler

from any_gold import AnyRawDataset, AnyVisionSegmentationDataset
from pixeltable import catalog, exprs

SEED = int(os.environ.get("SEED", 42))


def force_seed(seed: int = SEED) -> None:
    """
    Force the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_pxt_run_name(cfg: DictConfig) -> str:
    """
    Generate a unique name for the PixelTable based on the configuration.
    """
    if cfg.pixeltable.run_name is not None:
        return f"{cfg.pixeltable.dir_name}.{cfg.pixeltable.run_name}"

    dataset = cfg.dataset.args._target_.split(".")[-1]
    model = cfg.model.name
    split = cfg.dataset.args.split

    run_name = f"{cfg.pixeltable.dir_name}.{dataset}_{model}_{split}"
    if "category" in cfg.dataset.args:
        run_name += f"_{cfg.dataset.args.category}"

    return run_name


def get_pxt_table_name(
    cfg: DictConfig,
    run_name: str,
) -> str:
    """
    Generate a unique name for the PixelTable view based on the configuration.
    """
    view_name = (
        cfg.pixeltable.table_name
        if cfg.pixeltable.table_name is not None
        else f"seed_{cfg.pipeline.seed}_k_{cfg.pipeline.k_shots}_n_{cfg.pipeline.num_points}_amin_{cfg.pipeline.min_area}"
    )

    return f"{run_name}.{view_name}"


def get_pxt_table_path(
    cfg: DictConfig,
) -> str:
    """
    Generate the path to the PixelTable based on the configuration.
    """
    run_name = get_pxt_run_name(cfg)
    return get_pxt_table_name(cfg, run_name)


def format_for_pixeltable(value: Any, is_image: bool) -> Any:
    if isinstance(value, torch.Tensor):
        if is_image:
            return F.to_pil_image(value.cpu())
        else:
            if value.dim() == 0:
                return value.item()

            return value.numpy().astype(np.int64)
    else:
        return value


def import_any_dataset_to_pixeltable(
    dataset: AnyVisionSegmentationDataset,
    pxt_table: catalog.Table,
    label: str | None = None,
    num_samples: int | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> None:
    """Load a PyTorch dataset into a PixelTable.

    Args:
        dataset: The PyTorch dataset to import from.
        pxt_table: The PixelTable to import the data into.
        label: The name of the label column in the dataset.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses to use for data loading.
    """
    raw_dataset = AnyRawDataset(dataset)
    sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples)
    dataloader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )

    for batch in dataloader:
        for sample in batch:
            to_insert = {}
            for key, value in sample.items():
                value_for_pxt = format_for_pixeltable(value, key == "image")
                # if the label is a column name, rename this column `label`
                if key == label:
                    key = "label"

                # add missing column from table definition
                if key not in pxt_table.columns():
                    pxt_table.add_columns(
                        {
                            key: ColumnType.from_python_type(
                                type(value_for_pxt),
                                nullable_default=True,
                                allow_builtin_types=True,
                            )
                        }
                    )
                if key == "mask":
                    value_for_pxt = np.squeeze(value_for_pxt)
                to_insert[key] = value_for_pxt

            # missing label column mean the label should have the specified value
            if "label" not in to_insert:
                to_insert["label"] = label

            pxt_table.insert(**to_insert)


class PxtSAMSingleClickDatasetImporter(foud.LabeledImageDatasetImporter):
    """
    Implementation of a FiftyOne `DatasetImporter` that reads image data from a Pixeltable table.
    """

    def __init__(
        self,
        pxt_table: pxt.Table,
        image: exprs.Expr,
        masks: exprs.Expr,
        boxes: exprs.Expr,
        points: exprs.Expr,
        sam_logits: exprs.Expr | None,
        sam_masks: exprs.Expr | None,
        tmp_dir: str,
        dataset_dir: Optional[os.PathLike] = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir, shuffle=None, seed=None, max_samples=None
        )
        self._labels = {
            "ground_truth": (masks, fo.Segmentation),
            "boxes": (boxes, fo.Detection),
            "points": (points, fo.Keypoint),
        }
        if sam_logits is not None:
            self._labels["sam_logits"] = (sam_logits, fo.Heatmap)

        if sam_masks is not None:
            self._labels["sam_masks"] = (sam_masks, fo.Segmentation)

        self.tmp_dir = tmp_dir

        selection = [image] + [expr for expr, _ in self._labels.values()]

        assert image.col.is_stored
        selection.append(image.localpath)

        df = pxt_table.select(*selection)
        self._row_iter = (
            df._output_row_iterator()
        )  # iterator over the table rows, to be convered to FiftyOne samples

    def __next__(self) -> tuple[str, fo.ImageMetadata, dict[str, fo.Label]]:
        row = next(self._row_iter)
        img = row[0]
        file = row[-1]

        print(f"converting {file}")

        assert isinstance(img, Image), "Image data must be a PIL Image"
        metadata = fo.ImageMetadata(
            size_bytes=os.path.getsize(file),
            mime_type=puremagic.from_file(file, mime=True),
            width=img.width,
            height=img.height,
            filepath=file,
            num_channels=len(img.getbands()),
        )

        labels: dict[str, fo.Label] = {}
        for idx, (label_name, (_, label_cls)) in enumerate(self._labels.items()):
            label_data = row[idx + 1]  # +1 because the first column is the image
            if label_cls is fo.Segmentation:
                label_list = self._as_fo_segmentation(label_data, prefix=label_name)
            elif label_cls is fo.Detection:
                label_list = self._as_fo_detection(label_data, img.size)
            elif label_cls is fo.Keypoint:
                label_list = self._as_fo_keypoint(label_data, img.size)
            elif label_cls is fo.Heatmap:
                label_list = self._as_fo_heatmap_and_classification(label_data)

            if label_list is not None:
                if label_cls is not fo.Heatmap:
                    labels[label_name] = label_list
                else:
                    classification_list = []
                    heatmap_list = []
                    for heatmap, classification in label_list:
                        classification_list.append(classification)
                        map_path = f"{self.tmp_dir}/{heatmap.label}_{ObjectId()}.png"

                        cv2.imwrite(map_path, heatmap.map)  # Save the heatmap to a file
                        heatmap_list.append(
                            fo.Heatmap(
                                label=heatmap.label,
                                map_path=map_path,
                                range=heatmap.range,
                            )
                        )
                    labels["sam_logits"] = heatmap_list
                    labels["sam_iou"] = classification_list

        return (
            file,
            metadata,
            {
                label.label: label
                for label_list in labels.values()
                for label in label_list
            },
        )

    def _as_fo_segmentation(
        self, data: None | np.ndarray, prefix: str = "ground_truth"
    ) -> list[fo.Segmentation] | None:
        if data is None:
            return None

        if data.ndim == 3:
            # ground truth masks are in the shape (M, height, width)
            return [
                fo.Segmentation(
                    mask=mask,
                    label=f"{prefix}_{idx_mask}",
                )
                for idx_mask, mask in enumerate(data)
            ]
        else:
            return [
                fo.Segmentation(
                    mask=point_mask,
                    label=f"{prefix}_{idx_box}_{idx_point}",
                )
                for idx_box, box_mask in enumerate(data)
                for idx_point, point_mask in enumerate(box_mask)
            ]

    def _as_fo_detection(
        self, data: np.ndarray | None, img_size: tuple[int, int]
    ) -> list[fo.Detection] | None:
        if data is None:
            return None

        w, h = img_size
        return [
            fo.Detection(
                label=f"bounding_box_{idx_box}",
                bounding_box=[
                    box[0] / w,
                    box[1] / h,
                    (box[2] - box[0]) / w,
                    (box[3] - box[1]) / h,
                ],
            )
            for idx_box, box in enumerate(data)
        ]

    def _as_fo_keypoint(
        self, data: np.ndarray | None, img_size: tuple[int, int]
    ) -> list[fo.Keypoint] | None:
        if data is None:
            return None

        w, h = img_size
        return [
            fo.Keypoint(
                points=[(point[0] / w, point[1] / h)],  # x, y
                label=f"random_point_{idx_box}_{idx_point}",
            )
            for idx_box, box_points in enumerate(data)
            for idx_point, point in enumerate(box_points)
            if point.size > 0
        ]

    def _as_fo_heatmap_and_classification(
        self,
        data: np.ndarray | None,
    ) -> list[tuple[fo.Heatmap, fo.Classification]] | None:
        if data is None:
            return None

        def make_heatmap(arr: np.ndarray) -> np.ndarray:
            """Normalize the array to the range [0, 1]."""
            min_val = arr.min()
            max_val = arr.max()

            heatmap = (255 * ((arr - min_val) / (max_val - min_val + 1e-8))).astype(
                np.uint8
            )
            assert isinstance(heatmap, np.ndarray)
            return heatmap

        return [
            (
                fo.Heatmap(
                    map=make_heatmap(prediction[idx_logits]),
                    label=f"sam_logits_{idx_box}_{idx_point}_{idx_logits}",
                    range=[0, 255],
                ),
                fo.Classification(
                    label=f"sam_iou_{idx_box}_{idx_point}_{idx_logits}",
                    confidence=prediction[idx_logits + 3].max(),
                ),
            )
            for idx_box, box_predictions in enumerate(data)
            for idx_point, prediction in enumerate(box_predictions)
            for idx_logits in range(3)  # Assuming 3 logits
        ]

    @property
    def label_cls(self) -> dict[str, type[fo.Label]]:
        return {
            label_name: label_cls for label_name, (_, label_cls) in self._labels.items()
        }

    @property
    def has_dataset_info(self) -> bool:
        return False

    @property
    def has_image_metadata(self) -> bool:
        return True

    def setup(self) -> None:
        pass

    def get_dataset_info(self) -> dict:
        pass

    def close(self, *args: Any) -> None:
        pass
