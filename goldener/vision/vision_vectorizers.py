import torch
import torchvision

from goldener.vectorize import TensorVectorizer, Filter2DWithCount, FilterLocation


# add a torchvision transform allowing to patchify a mask and
# then use the ratio of non zero pixel as value
class PatchifyImageMask(torchvision.transforms.v2.Transform):
    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x is a mask of shape (B, H, W)
        B, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            "H and W must be divisible by patch_size"
        )

        x = x.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        x = x.contiguous().view(B, -1, self.patch_size * self.patch_size)
        return (
            (x.sum(dim=-1) > 0).float() / 16 * 16
        )  # return 1 if any pixel in the patch is non-zero, else return 0


def get_vit_class_token_vectorizer() -> TensorVectorizer:
    return TensorVectorizer(
        keep=Filter2DWithCount(
            filter_count=1, keep=True, filter_location=FilterLocation.START
        ),
        channel_pos=2,
    )


def get_vit_prefix_tokens_vectorizer(n_prefixes: int = 5) -> TensorVectorizer:
    return TensorVectorizer(
        keep=Filter2DWithCount(
            filter_count=n_prefixes, keep=True, filter_location=FilterLocation.START
        ),
        channel_pos=2,
    )
