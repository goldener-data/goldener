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
