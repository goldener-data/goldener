import torch

from goldener.vision.vectorizers import (
    get_vit_class_token_vectorizer,
    get_vit_prefix_tokens_vectorizer,
    get_vit_patch_tokens_vectorizer,
)
from goldener.vectorize import TensorVectorizer, Filter2DWithCount, FilterLocation


class TestVitVectorizerHelpers:
    def make_tensor(self, shape=(2, 3, 4)):
        # Simple tensor with increasing integers for deterministic checks
        return torch.arange(shape[0] * shape[1] * shape[2]).reshape(shape)

    def test_get_vit_class_token_vectorizer(self):
        x = self.make_tensor()
        vec = get_vit_class_token_vectorizer()

        out = vec.vectorize(x)
        # With keep filter_count=1 and channel_pos=2, expect one vector per batch
        assert out.vectors.shape[0] == x.shape[0]
        # Features dimension is the flattened non-channel dims after moving channel to dim 1
        assert out.vectors.shape[1] == x.shape[2]
        # batch_indices should enumerate batches
        assert torch.equal(out.batch_indices, torch.arange(x.shape[0]))

    def test_get_vit_prefix_tokens_vectorizer_default(self):
        x = self.make_tensor((2, 6, 4))  # C=6, will keep first 5 prefixes by default
        vec = get_vit_prefix_tokens_vectorizer(n_prefixes=5)
        assert isinstance(vec, TensorVectorizer)

        out = vec.vectorize(x)
        # Expect 5 vectors per batch (5 prefixes), flattened over sequence dimension
        assert out.vectors.shape[0] == x.shape[0] * 5
        assert out.vectors.shape[1] == x.shape[2]
        # batch indices must repeat each batch index 5 times
        expected_batches = torch.arange(x.shape[0]).repeat_interleave(5)
        assert torch.equal(out.batch_indices, expected_batches)

    def test_get_vit_patch_tokens_vectorizer_only_remove_prefixes(self):
        # Sequence length 6 along channel dim => after removing first 2, expect 4 per batch
        x = self.make_tensor((2, 6, 4))
        vec = get_vit_patch_tokens_vectorizer(n_prefixes=2, n_random=None)

        out = vec.vectorize(x)
        # 4 patch tokens per batch
        assert out.vectors.shape[0] == x.shape[0] * 4
        assert out.vectors.shape[1] == x.shape[2]

    def test_get_vit_patch_tokens_vectorizer_only_random(self):
        # No prefixes removed, but randomly keep 3 tokens per batch from 6
        x = self.make_tensor((2, 6, 4))
        # To make test deterministic, we pass a generator with fixed seed
        rand_filter = Filter2DWithCount(
            filter_count=3,
            filter_location=FilterLocation.RANDOM,
            keep=True,
            generator=torch.Generator().manual_seed(0),
        )
        # Manually build vectorizer to mirror helper behavior when only random is set
        vec = TensorVectorizer(random=rand_filter, channel_pos=2)

        out = vec.vectorize(x)
        assert out.vectors.shape[0] == x.shape[0] * 3
        assert out.vectors.shape[1] == x.shape[2]

    def test_get_vit_patch_tokens_vectorizer_remove_and_random(self):
        # Sequence length 8: remove 2 prefixes, then randomly keep 3 of remaining 6
        x = self.make_tensor((2, 8, 4))
        generator = torch.Generator().manual_seed(0)
        # Build vectorizer like get_vit_patch_tokens_vectorizer but with deterministic random filter
        remove = Filter2DWithCount(
            filter_count=2,
            keep=False,
            filter_location=FilterLocation.START,
        )
        rand = Filter2DWithCount(
            filter_count=3,
            keep=True,
            filter_location=FilterLocation.RANDOM,
            generator=generator,
        )
        vec = TensorVectorizer(remove=remove, random=rand, channel_pos=2)

        out = vec.vectorize(x)
        # After removing 2 prefixes, there are 6 tokens per batch; we keep 3 of them
        assert out.vectors.shape[0] == x.shape[0] * 3
        assert out.vectors.shape[1] == x.shape[2]
