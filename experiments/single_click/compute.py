import cv2
import numpy as np
from PIL.Image import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor


def extract_connected_components_from_binary_mask(
    mask: np.ndarray, min_area: int = 1
) -> np.ndarray | None:
    """Find the connected components in the binary masks.

    A connected component is a set of pixels that are connected together.

    Args:
        mask: binary masks with shape (H, W).
        min_area: The minimum area of a connected component to be considered valid.

    Returns:
        An optional array of shape (M, H, W) corresponding to the connected component masks.
        None if no valid connected components (area > min_area) are found or if the input mask is empty.

    Raises:
        ValueError: If mask is not a 2D array (H, W).
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array (H, W).")

    if (mask == 0).all():
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )

    valid_connected_components = [
        (labels == label_idx).astype(np.uint8)
        for label_idx in range(1, num_labels)  # Exclude background label (0)
        if stats[label_idx][4] >= min_area
    ]

    if not valid_connected_components:
        return None

    return np.stack(valid_connected_components, axis=0)


def extract_bounding_boxes_from_connected_components(
    connected_components: np.ndarray | None,
) -> np.ndarray | None:
    """Calculate bounding boxes for the connected components.

    Args:
        connected_components: Optional array of connected component masks with shape (M, H, W),
        where M is the number of different connected components. All HxW masks
        are binary masks with 1s for the connected components and 0s elsewhere.

    Returns:
        An optional array of bounding boxes with shape (M, 4), where each box is represented
        with coordinates (x1, y1, x2, y2) corresponding to the left(x)-top(y) and right(x)-bottom(y) corners.
        If connected_components is None, returns None.

    Raises:
        ValueError: If connected_components is not a 3D array (M, H, W).
        ValueError: If connected_components contains empty connected components or multiple areas of connected components.
    """
    if connected_components is None:
        return None
    elif connected_components.ndim != 3:
        raise ValueError("Input connected_components must be a 3D array (M, H, W).")

    boxes = []

    for connected_component in connected_components:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(connected_component)
        if num_labels == 1:
            raise ValueError(
                "empty connected component found, cannot find bounding box."
            )
        elif num_labels > 2:
            raise ValueError(
                "connected component should have only one area of connected components, "
                "but found multiple ones."
            )

        boxes.append(
            np.array(
                (
                    stats[1, cv2.CC_STAT_LEFT],
                    stats[1, cv2.CC_STAT_TOP],
                    stats[1, cv2.CC_STAT_LEFT] + stats[1, cv2.CC_STAT_WIDTH],
                    stats[1, cv2.CC_STAT_TOP] + stats[1, cv2.CC_STAT_HEIGHT],
                ),  # background label = 0
                dtype=np.uint32,
            )
        )

    return np.stack(boxes, axis=0)


def extract_random_points_from_connected_components(
    connected_components: np.ndarray | None,
    num_points: int = 1,
) -> np.ndarray | None:
    """Sample random points within the connected components.

    Args:
        connected_components: Optional array of connected component masks with shape (M, H, W),
        where M is the number of connected components. All HxW masks
        are binary masks with 1s for the connected components and 0s elsewhere.
        num_points: Number of random points to sample from each connected component.

    Returns:
        An optional array of shape (M, num_points, 2), where each point is represented
        as (x, y) coordinates and randomly sampled from non zero pixel in each connected components.
        If connected_components is None, returns None.

    Raises:
        ValueError: If num_points is less than 1.
        ValueError: If connected_components is not a 3D array (M, H, W).
        ValueError: If connected_components contains empty connected component or
        if the number of points requested is greater than the number of points in the connected component.
    """
    if num_points < 1:
        raise ValueError("num_points must be at least 1.")

    if connected_components is None:
        return None
    elif connected_components.ndim != 3:
        raise ValueError("Input connected_components must be a 3D array (M, H, W).")

    points = []

    for connected_component in connected_components:
        indices = np.argwhere(connected_component)

        if indices.size == 0:
            raise ValueError("empty connected component found, cannot sample points.")
        elif indices.shape[0] < num_points:
            raise ValueError(
                f"connected component has only {indices.shape[0]} points, "
                f"but requested {num_points} points."
            )

        selected_indices = np.random.choice(
            indices.shape[0], size=num_points, replace=False
        )
        points.append(
            np.array([[point[1], point[0]] for point in indices[selected_indices]])
        )

    return np.stack(points, axis=0)


def predict_sam_logits_from_single_click(
    model: SAM2ImagePredictor,
    image: Image,
    boxes: np.ndarray | None,
    points: np.ndarray | None,
) -> np.ndarray | None:
    """Predict logits and iou predictions after a single click using the SAM model.

    Args:
        model: The SAM model to use.
        image: The input image to process.
        boxes: An optional array of bounding boxes with shape (M, 4), where each box is represented
        with coordinates (x1, y1, x2, y2) corresponding to the left(x)-top(y) and right(x)-bottom(y) corners.
        Multiple SAM logits might be predicted for each box.
        points: An optional array of shape (M, num_points, 2), where each point is represented
        as (x, y) coordinates. A SAM logits will be predicted for each point.

    Returns:
        An optional array of shape (M, num_points, 6, H, W), where the logits of shape (H, W) for each point is the
        concatenation of the 3 mask logits and 3 IoU predictions coming from SAM. The IOU predictions
        are converted as map with the same value for all pixel in order to return a single array (pixeltable constraint).
        If boxes is None, returns None.

    Raises:
        ValueError: If boxes is None and points is not None.
        ValueError: If boxes is not None and points is None.
        ValueError: If boxes is not a 2D array with shape (M, 4).
        ValueError: If points is not a 3D array with shape (M, num_points, 2).
        ValueError: If boxes and points do not have the same number of elements (M).
    """
    if boxes is None:
        if points is not None:
            raise ValueError("If boxes is None, points should also be None.")
        return None
    elif points is None:
        raise ValueError("If boxes is not None, points should not be None.")

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Input boxes must be a 2D array with shape (M, 4).")
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError(
            "Input points must be a 3D array with shape (M, num_points, 2)."
        )
    if boxes.shape[0] != points.shape[0]:
        raise ValueError(
            "Input boxes and points must have the same number of elements (M)."
        )
    # sam applies preprocessing and compute feature maps
    model.set_image(image)

    masks = []

    for box, box_points in zip(boxes, points, strict=True):
        box_masks = []
        labels = np.ones((box_points.shape[0], 1), dtype=np.int64)

        for point, label in zip(box_points, labels):
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
                )  # Concatenate masks and iou predictions in the same array to store in a pixeltable
            )

        masks.append(np.stack(box_masks, axis=0))

    return np.stack(masks, axis=0)


def threshold_single_click_sam_logits(
    sam_logits: np.ndarray | None,
    threshold: float = 0.0,
) -> np.ndarray | None:
    """Threshold the SAM logits to create binary masks.

    Args:
        sam_logits: An optional array of shape (M, num_points, 6, H, W), where the logits of shape (H, W)
        for each point is the concatenation of the 3 mask logits and 3 IoU predictions coming from SAM.
        The IOU predictions are converted as map with the same value for all pixels.
        threshold: The threshold to apply to the logits to create binary masks.

    Returns: An optional array of shape (M, num_points, H, W) containing the binary masks. Among the SAM logits,
    the choice between the 3 masks is made based on the best IoU prediction. Then these logits are thresholded
    to create binary masks.

    Raises:
        ValueError: If sam_logits is not a 5D array with shape (M, num_points, 6, H, W).
    """
    if sam_logits is None:
        return None

    if sam_logits.ndim != 5 or sam_logits.shape[2] != 6:
        raise ValueError(
            "Input sam_logits must be a 5D array with shape (M, num_points, 6, H, W)."
        )

    sam_masks = []

    for box_logits in sam_logits:
        box_masks = []
        for point_logits in box_logits:
            # ious are the 3 last channels filled with all the same value (IOU prediction for the corresponding mask)
            ious = point_logits[-3:, 0, 0]
            best_iou_index = np.argmax(ious)  # Get the index of the best IoU prediction
            box_masks.append(
                (point_logits[best_iou_index] > threshold).astype(
                    np.uint8
                )  # Apply threshold to the best IoU mask
            )
        sam_masks.append(np.stack(box_masks, axis=0))

    return np.stack(sam_masks, axis=0)


def compute_ious_from_sam_masks_and_connected_components(
    connected_components: np.ndarray | None,
    predicted_masks: np.ndarray | None,
) -> np.ndarray | None:
    """Calculate the Intersection over Union (IoU) for the connected components and predicted masks.

    Args:
        connected_components: An optional array of shape (M, H, W) corresponding to the connected component masks.
        predicted_masks: An optional array of shape (M, num_points, H, W) containing the binary masks
        obtained from the SAM logits.

    Returns:
        An optional array of shape (M, num_points) containing the IoU values for each sampled points
        for each connected component. If connected_components is None, returns None.

    Raises:
        ValueError: If connected_components is None and predicted_masks is not None.
        ValueError: If connected_components is not None and predicted_masks is None.
        ValueError: If connected_components is not a 3D array (M, H, W).
        ValueError: If predicted_masks is not a 4D array (M, num_points, H, W).
        ValueError: If connected_components and predicted_masks do not have the same number of elements (M).
    """
    if connected_components is None:
        if predicted_masks is not None:
            raise ValueError(
                "If connected_components is None, predicted_masks should also be None."
            )
        return None
    elif predicted_masks is None:
        raise ValueError(
            "If connected_components is not None, predicted_masks should not be None."
        )

    if connected_components.ndim != 3:
        raise ValueError("Input connected_components must be a 3D array (M, H, W).")
    if predicted_masks.ndim != 4:
        raise ValueError(
            "Input predicted_masks must be a 4D array (M, num_points, H, W)."
        )
    if connected_components.shape[0] != predicted_masks.shape[0]:
        raise ValueError(
            "Input connected_components and predicted_masks must have the same number of elements (M)."
        )

    ious = []

    for box_mask, box_predicted_masks in zip(connected_components, predicted_masks):
        box_ious = []
        for point_predicted_mask in box_predicted_masks:
            intersection = np.logical_and(box_mask, point_predicted_mask)
            union = np.logical_or(box_mask, point_predicted_mask)
            assert (
                union > 0
            ).any(), "Union of masks should be greater than 0 to avoid division by zero"
            iou = (
                intersection.sum() / union.sum()
            )  # Calculate IoU for the connected component and the predicted mask
            box_ious.append(np.array(iou, dtype=np.float32))

        ious.append(np.stack(box_ious, axis=0))

    return np.stack(ious, axis=0)
