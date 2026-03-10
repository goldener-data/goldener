import torch
import torch.nn.functional as F


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes cosine distance between each pair of the two collections of row vectors.

    Args:
        x1: A tensor of shape (n, d) representing n vectors of dimension d.
        x2: A tensor of shape (m, d) representing m vectors of dimension d.

    Returns:
        A tensor of shape (n, m) where the entry at (i, j) is the cosine distance between x1[i] and x2[j].
    """
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("Input tensors must be 2D")

    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input tensors must have the same number of channels")

    x_norm = F.normalize(x1, p=2, dim=-1)
    y_norm = F.normalize(x2, p=2, dim=-1)

    similarity = (x_norm @ y_norm.T).clamp(min=-1.0, max=1.0)

    return 1 - similarity
