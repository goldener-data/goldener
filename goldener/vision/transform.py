from typing import Any

import torch
from torchvision.transforms.v2 import Transform


class PatchifyImageMask(Transform):
    """Patchify a binary mask in order to align it with the tokens of a ViT model.

    Attributes:
        patch_size: The size of the patches to create.
        match_ratio: The ratio of non zero pixels in a patch to consider it as a valid token for the class.
    """

    def __init__(self, patch_size: int = 16, match_ratio: float = 0.5) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        self.patch_size = patch_size

        if not 0.0 <= match_ratio <= 1.0:
            raise ValueError("match_ratio must be between 0 and 1")

        self.match_ratio = match_ratio

    def transform(self, inpt: Any, params: dict[str, Any] | None = None) -> Any:
        """Patchify a binary mask in order to align it with the tokens of a ViT model.

        Args:
            inpt: A binary mask tensor of shape (B, 1, H, W) where B is the batch size, H and W are the height and width of the image.
            params: A dictionary of parameters (not used in this transform).

        Returns:
            A tensor of shape (B, N) where N is the number of patches (H // patch_size) * (W // patch_size)
            and each value is 1 if the ratio of non zero pixels in the patch is greater than match_ratio, otherwise 0.
        """
        if inpt.ndim != 4:
            raise ValueError("Input tensor must have shape (B, C, H, W)")

        B, C, H, W = inpt.shape
        if C != 1:
            raise ValueError("Input mask must have a single channel (C=1)")

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("H and W must be divisible by patch_size")

        if not torch.all((inpt == 0) | (inpt == 1)):
            raise ValueError("Input must be a binary mask with values 0 or 1")

        pixels_in_patch = self.patch_size * self.patch_size

        # Patchify over spatial dimensions (H, W) while keeping batch and channel dimensions
        patches = inpt.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # patches shape: (B, 1, H//ps, W//ps, ps, ps)
        patches = patches.contiguous().view(B, -1, pixels_in_patch)
        # compute the ratio of non zero pixels in each patch
        ratios = patches.sum(dim=-1) / pixels_in_patch

        return (ratios > self.match_ratio).float()
