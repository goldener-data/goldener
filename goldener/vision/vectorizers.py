import torch

from goldener.vectorize import TensorVectorizer, Filter2DWithCount, FilterLocation


def get_vit_class_token_vectorizer() -> TensorVectorizer:
    """Get a TensorVectorizer that keeps only the class token from a ViT model."""
    return TensorVectorizer(
        keep=Filter2DWithCount(
            filter_count=1, keep=True, filter_location=FilterLocation.START
        ),
        channel_pos=2,
    )


def get_vit_prefix_tokens_vectorizer(n_prefixes: int = 5) -> TensorVectorizer:
    """Get a TensorVectorizer that keeps the first n_prefixes tokens from a ViT model.

    Args:
        n_prefixes: The number of prefix tokens to keep (default is 5).
    """
    if n_prefixes <= 0:
        raise ValueError("n_prefixes must be a positive integer")

    return TensorVectorizer(
        keep=Filter2DWithCount(
            filter_count=n_prefixes, keep=True, filter_location=FilterLocation.START
        ),
        channel_pos=2,
    )


def get_vit_patch_tokens_vectorizer(
    n_prefixes: int | None = 5,
    n_random: int | None = None,
    generator: torch.Generator | None = None,
) -> TensorVectorizer:
    """Get a TensorVectorizer that keeps the patch tokens from a ViT model

    It optionally removes the first n_prefixes tokens and keeps n_random random tokens.

    Args:
       n_prefixes: The number of prefix tokens to remove from the start (default is 5).
           If None, no prefixes are removed.
       n_random: The number of random tokens to keep after removing prefixes
           (default is None, meaning no random tokens are kept).
           If None, no random filtering is applied.
       generator: An optional torch.Generator for reproducibility when n_random is set.
    """
    if n_prefixes is not None and n_prefixes <= 0:
        raise ValueError("n_prefixes must be a positive integer or None")
    if n_random is not None and n_random <= 0:
        raise ValueError("n_random must be a positive integer or None")

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
                generator=generator,
            )
            if n_random is not None
            else None
        ),
        channel_pos=2,
    )
