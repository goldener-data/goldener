import torch
import torch.nn.functional as F


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes cosine distance between each pair of the two collections of row vectors."""
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("Input tensors must be 2D")

    x_norm = F.normalize(x1, p=2, dim=-1)
    y_norm = F.normalize(x2, p=2, dim=-1)

    similarity = (x_norm @ y_norm.T).clamp(min=-1.0, max=1.0)

    return 1 - similarity
