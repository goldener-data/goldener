from typing import Optional

from PIL.Image import Image

import pixeltable as pxt

from experiments.single_click.model import sam_cache
from experiments.single_click.compute import (
    extract_connected_components_from_binary_mask,
    extract_bounding_boxes_from_connected_components,
    extract_random_points_from_connected_components,
    predict_sam_logits_from_single_click,
    threshold_single_click_sam_logits,
    compute_ious_from_sam_masks_and_connected_components,
)


@pxt.udf
def connected_components(
    mask: pxt.Array, min_area: int
) -> Optional[pxt.Array[pxt.Int]]:
    """Find the connected components in the binary masks.

    see `utils.extract_connected_components_from_mask` for more details.
    """
    return extract_connected_components_from_binary_mask(mask=mask, min_area=min_area)


@pxt.udf
def bounding_boxes(connected_components: Optional[pxt.Array]) -> Optional[pxt.Array]:
    """Calculate bounding boxes for the connected components.

    see `utils.extract_bounding_boxes_from_connected_components` for more details.
    """
    return extract_bounding_boxes_from_connected_components(
        connected_components=connected_components
    )


@pxt.udf
def random_points(
    connected_components: Optional[pxt.Array],
    num_points: int = 1,
) -> Optional[pxt.Array]:
    """Sample random points within the connected components.

    See `utils.extract_random_points_from_connected_components` for more details.
    """
    return extract_random_points_from_connected_components(
        connected_components=connected_components,
        num_points=num_points,
    )


@pxt.udf
def sam_logits_from_single_click(
    model_id: str,
    image: Image,
    boxes: Optional[pxt.Array],
    points: Optional[pxt.Array],
) -> Optional[pxt.Array]:
    """Predict logits and iou predictions after a single click using the SAM model.

    Args:
        model_id: The identifier of the SAM model to use.
        see `utils.predict_sam_logits_from_single_click for other arguments

    See `utils.predict_sam_logits_from_single_click` for more details.
    """
    if model_id not in sam_cache:
        raise ValueError(
            f"Model with id {model_id} is not loaded. Please load the model first."
        )

    model = sam_cache[model_id]

    return predict_sam_logits_from_single_click(
        model=model,
        image=image,
        boxes=boxes,
        points=points,
    )


@pxt.udf
def masks_from_sam_logits(
    sam_logits: Optional[pxt.Array],
    threshold: float = 0.0,
) -> Optional[pxt.Array]:
    """Threshold the SAM logits to create binary masks.

    See `utils.threshold_single_click_sam_logits` for more details.
    """
    return threshold_single_click_sam_logits(
        sam_logits=sam_logits,
        threshold=threshold,
    )


@pxt.udf
def segmentation_ious(
    connected_components: Optional[pxt.Array],
    predicted_masks: Optional[pxt.Array],
) -> Optional[pxt.Array]:
    """Calculate the Intersection over Union (IoU) for the connected components and predicted masks.

    see `utils.compute_ious_from_sam_masks_and_connected_components` for more details.
    """
    return compute_ious_from_sam_masks_and_connected_components(
        connected_components=connected_components,
        predicted_masks=predicted_masks,
    )
