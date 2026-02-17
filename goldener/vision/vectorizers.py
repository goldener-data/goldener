import torch
from torchvision.transforms.v2 import Transform

from goldener.vectorize import TensorVectorizer, Filter2DWithCount, FilterLocation


# add a torchvision transform allowing to patchify a mask and
# then use the ratio of non zero pixel as value
class PatchifyImageMask(Transform):
    """Patchify a binary mask in order to align it with the tokens of a ViT model.

    Attributes:
        patch_size: The size of the patches to create.
        match_ratio: The ratio of non zero pixels in a patch to consider it as a valid token for the class.
    """

    def __init__(self, patch_size: int = 16, match_ratio: float = 0.5) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.match_ratio = match_ratio

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify a binary mask in order to align it with the tokens of a ViT model.

        Args:
            x: A binary mask tensor of shape (B, 1, H, W) where B is the batch size, H and W are the height and width of the image.

        Returns:
            A tensor of shape (B, N) where N is the number of patches (H // patch_size) * (W // patch_size)
            and each value is 1 if the ratio of non zero pixels in the patch is greater than match_ratio, otherwise 0.
        """
        if x.ndim != 4:
            raise ValueError("Input tensor must have shape (B, H, W)")

        B, C, H, W = x.shape
        if C != 1:
            raise ValueError("Input mask must have a single channel (C=1)")

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("H and W must be divisible by patch_size")

        pixels_in_patch = self.patch_size * self.patch_size

        # Patchify over spatial dimensions (H, W) while keeping batch and channel dimensions
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # patches shape: (B, 1, H//ps, W//ps, ps, ps)
        patches = patches.contiguous().view(B, -1, pixels_in_patch)
        # compute the ratio of non zero pixels in each patch
        ratios = patches.sum(dim=-1) / pixels_in_patch

        return (ratios > self.match_ratio).float()


def get_vit_class_token_vectorizer() -> TensorVectorizer:
    """Get a TensorVectorizer that keeps only the class token from a ViT model."""
    return TensorVectorizer(
        keep=Filter2DWithCount(
            filter_count=1, keep=True, filter_location=FilterLocation.START
        ),
        channel_pos=2,
    )


def get_vit_prefix_tokens_vectorizer(n_prefixes: int = 5) -> TensorVectorizer:
    """Get a TensorVectorizer that keeps the first n_prefixes tokens from a ViT model."""
    return TensorVectorizer(
        keep=Filter2DWithCount(
            filter_count=n_prefixes, keep=True, filter_location=FilterLocation.START
        ),
        channel_pos=2,
    )


def get_vit_patch_tokens_vectorizer(
    n_prefixes: int | None = 5, n_random: int | None = None
) -> TensorVectorizer:
    """Get a TensorVectorizer that keeps the patch tokens from a ViT model, optionally removing the first n_prefixes tokens and keeping n_random random tokens."""
    return TensorVectorizer(
        remove=(
            Filter2DWithCount(
                filter_count=n_prefixes,
                keep=False,
                filter_location=FilterLocation.START,
            )
            if n_prefixes is not None
            else None
        ),
        random=(
            Filter2DWithCount(
                filter_count=n_random,
                keep=True,
                filter_location=FilterLocation.RANDOM,
            )
            if n_random is not None
            else None
        ),
        channel_pos=2,
    )
