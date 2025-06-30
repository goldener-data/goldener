from typing import Optional

import cv2
import numpy as np
from PIL.Image import Image

import pixeltable as pxt

from experiments.single_click.load import sam_cache


@pxt.udf(is_property=True)
def image_size(self: Image) -> tuple[int, int]:
    return self.size


@pxt.udf
def connected_components(masks: pxt.Array, min_area: int) -> pxt.Array[pxt.Int]:
    connected_component_masks = []
    for mask in masks:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8)
        )
        connected_component_masks.extend(
            [
                (labels == label_idx).astype(np.uint8)
                for label_idx in range(1, num_labels)  # Exclude background label (0)
                if stats[label_idx][4] >= min_area
            ]
        )

    return (
        np.stack(connected_component_masks, axis=0).astype(np.int64)
        if connected_component_masks
        else masks
    )


@pxt.udf
def bounding_boxes(masks: pxt.Array) -> pxt.Array:
    boxes = []
    for mask in masks:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8)
        )
        if num_labels > 1:
            boxes.append(
                [
                    (
                        stats[label_idx, cv2.CC_STAT_LEFT],
                        stats[label_idx, cv2.CC_STAT_TOP],
                        stats[label_idx, cv2.CC_STAT_LEFT]
                        + stats[label_idx, cv2.CC_STAT_WIDTH],
                        stats[label_idx, cv2.CC_STAT_TOP]
                        + stats[label_idx, cv2.CC_STAT_HEIGHT],
                    )  # x, y, width, height
                    for label_idx in range(
                        1, num_labels
                    )  # Exclude background label (0)
                ]
            )

    return (
        np.array(boxes, dtype=np.int64)
        if boxes
        else np.empty((0, 0, 4), dtype=np.int64)
    )


@pxt.udf
def random_points(
    masks: pxt.Array,
    boxes: Optional[pxt.Array],
    num_points: int = 1,
) -> pxt.Array:
    points: list[Optional[pxt.Array]] = []

    if boxes.size == 0:
        return np.empty((0, 0, 2), dtype=np.int64)

    for mask_idx, mask in enumerate(masks):
        for box in boxes[mask_idx]:
            # Ensure the box is within the mask bounds
            x1, y1, x2, y2 = box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(mask.shape[1], x2)
            y2 = min(mask.shape[0], y2)

            only_box = np.zeros_like(mask, dtype=np.int64)
            only_box[y1:y2, x1:x2] = 1
            only_box_mask = mask * only_box  # Apply the box to the mask

            if (only_box_mask == 0).all():
                # If the mask is empty, append an empty array
                points.append(np.empty((0, 2), dtype=np.int32))
            else:
                indices = np.argwhere(only_box_mask)
                selected_indices = np.random.choice(
                    indices.shape[0], size=num_points, replace=False
                )
                points.append(
                    np.array(
                        [[point[1], point[0]] for point in indices[selected_indices]]
                    )
                )

    return (
        np.array(points, dtype=np.int64)
        if points
        else np.empty((0, 0, 2), dtype=np.int64)
    )


@pxt.udf
def predict_with_sam(
    model_id: str,
    image: Image,
    boxes: Optional[pxt.Array],
    points: Optional[pxt.Array],
) -> pxt.Array:
    if model_id not in sam_cache:
        raise ValueError(
            f"Model with id {model_id} is not loaded. Please load the model first."
        )

    model = sam_cache[model_id]

    if boxes.size == 0:
        return np.zeros((0, 5, 6, *image.size), dtype=np.float32)

    masks = []
    assert boxes is not None and points is not None

    model.set_image(image)

    for box, point_box in zip(boxes, points, strict=True):
        box_masks = []
        labels = np.ones((point_box.shape[0], 1), dtype=np.int64)
        for point, label in zip(point_box, labels):
            sam_masks, iou_predictions, _ = model.predict(
                box=box,
                point_coords=point[np.newaxis, ...],
                point_labels=label,
                return_logits=True,
            )

            box_masks.append(
                np.concatenate(
                    [
                        sam_masks,
                        (
                            np.zeros_like(sam_masks)
                            + iou_predictions.reshape(*iou_predictions.shape, 1, 1)
                        ),
                    ],
                    axis=0,
                )  # Concatenate masks and iou predictions in the same array
            )
        masks.append(np.stack(box_masks, axis=0))

    stacked = np.stack(masks, axis=0)

    return stacked if stacked.ndim > 4 else stacked[np.newaxis, ...]


@pxt.udf
def mask_prediction_from_sam_logits(
    sam_logits: pxt.Array,
    threshold: float = 0.0,
) -> pxt.Array:
    """
    Threshold the SAM logits to create binary masks.
    """
    sam_masks = []
    for box_logits in sam_logits:
        for point_logits in box_logits:
            ious = point_logits[
                -3:, 0, 0
            ]  # ious are the 3 last channels filled with all the same value (IOU prediction for the corresponding mask)
            best_iou_index = np.argmax(ious)  # Get the index of the best IoU prediction
            sam_masks.append(
                (point_logits[best_iou_index] > threshold).astype(
                    np.float32
                )  # Apply threshold to the best IoU mask
            )

    stacked = (
        np.stack(sam_masks, axis=0)
        if sam_masks
        else np.zeros((1, 0, *sam_logits.shape[-2:]), dtype=np.float32)
    )

    return stacked if stacked.ndim > 4 else stacked[np.newaxis, ...]
