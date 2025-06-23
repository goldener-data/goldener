import os
from typing import Optional, Any

import cv2
import numpy as np
import puremagic
import torch
from PIL.Image import Image
from hydra.utils import instantiate
from omegaconf import DictConfig

from huggingface_hub import hf_hub_download
import fiftyone as fo
import fiftyone.utils.data as foud
from pixeltable import exprs

from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
from sam2.sam2_image_predictor import SAM2ImagePredictor

import pixeltable as pxt


def load_sam2_image_predictor_from_huggingface(
    model_cfg: DictConfig,
    device: str = "cuda",
    mode: str = "eval",
) -> SAM2ImagePredictor:
    hf_id = model_cfg.hf_id

    ckpt_path = hf_hub_download(
        repo_id=hf_id, filename=HF_MODEL_ID_TO_FILENAMES[hf_id][1]
    )
    model = instantiate(model_cfg.config.model, _recursive_=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Missing or unexpected keys in state dict to SAM 2 checkpoint."
        )

    model = model.to(device)
    if mode == "eval":
        model.eval()

    return SAM2ImagePredictor(sam_model=model)


sam_cache: dict[str, SAM2ImagePredictor] = {}


class PxtSAMDatasetImporter(foud.LabeledImageDatasetImporter):
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
        predictions: exprs.Expr,
        tmp_dir: str,
        dataset_dir: Optional[os.PathLike] = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir, shuffle=None, seed=None, max_samples=None
        )
        self._labels = {
            "masks": (masks, fo.Segmentation),
            "boxes": (boxes, fo.Detection),
            "points": (points, fo.Keypoint),
            "predictions": (predictions, fo.Heatmap),
        }
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
                label_list = self._as_fo_segmentation(label_data)
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
                    heamtap_list = []
                    for heatmap, classification in label_list:
                        classification_list.append(classification)
                        map_path = f"{self.tmp_dir}/{heatmap.label}.png"

                        cv2.imwrite(map_path, heatmap.map)  # Save the heatmap to a file
                        heamtap_list.append(
                            fo.Heatmap(
                                label=heatmap.label,
                                map_path=map_path,
                                range=heatmap.range,
                            )
                        )
                    labels["sam_logits"] = heamtap_list
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

    def _as_fo_segmentation(self, data: pxt.Array) -> list[fo.Segmentation] | None:
        if (data == 0).all():
            return None

        return [
            fo.Segmentation(
                mask=mask,
                label="ground_truth",
            )
            for mask in data
        ]

    def _as_fo_detection(
        self, data: pxt.Array, img_size: tuple[int, int]
    ) -> list[fo.Detection] | None:
        if data.size == 0:
            return None

        w, h = img_size
        return [
            fo.Detection(
                label=f"bounding_box_{idx_boxes}_{idx_box}",
                bounding_box=[
                    box[1] / w,
                    box[0] / h,
                    (box[3] - box[1]) / w,
                    (box[2] - box[0]) / h,
                ],
            )
            for idx_boxes, boxes in enumerate(data)
            for idx_box, box in enumerate(boxes)
        ]

    def _as_fo_keypoint(
        self, data: pxt.Array, img_size: tuple[int, int]
    ) -> list[fo.Keypoint] | None:
        if data.size == 0:
            return None

        w, h = img_size
        return [
            fo.Keypoint(
                points=[(point[1] / w, point[0] / h)],  # x, y
                label=f"random_point_{idx_box}_{idx_point}",
            )
            for idx_box, box_points in enumerate(data)
            for idx_point, point in enumerate(box_points)
            if point.size > 0
        ]

    def _as_fo_heatmap_and_classification(
        self, data: pxt.Array
    ) -> list[tuple[fo.Heatmap, fo.Classification]] | None:
        if (data == 0).all():
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
