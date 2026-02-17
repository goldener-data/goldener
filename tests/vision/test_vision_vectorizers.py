import torch

from goldener.vision.vectorizers import (
    get_vit_class_token_vectorizer,
    get_vit_prefix_tokens_vectorizer,
    get_vit_patch_tokens_vectorizer,
)
from goldener.vectorize import TensorVectorizer, Filter2DWithCount, FilterLocation


class TestVitVectorizerHelpers:
    def make_tensor(self, shape=(2, 3, 4)):
        return torch.arange(shape[0] * shape[1] * shape[2]).reshape(shape)

    def test_get_vit_class_token_vectorizer(self):
        x = self.make_tensor()
        vec = get_vit_class_token_vectorizer()

        out = vec.vectorize(x)
        assert out.vectors.shape[0] == x.shape[0]
        assert out.vectors.shape[1] == x.shape[2]
        assert torch.equal(out.batch_indices, torch.arange(x.shape[0]))

    def test_get_vit_prefix_tokens_vectorizer_default(self):
        x = self.make_tensor((2, 6, 4))
        vec = get_vit_prefix_tokens_vectorizer(n_prefixes=5)
        assert isinstance(vec, TensorVectorizer)

        out = vec.vectorize(x)
        assert out.vectors.shape[0] == x.shape[0] * 5
        assert out.vectors.shape[1] == x.shape[2]
        expected_batches = torch.arange(x.shape[0]).repeat_interleave(5)
        assert torch.equal(out.batch_indices, expected_batches)

    def test_get_vit_patch_tokens_vectorizer_only_remove_prefixes(self):
        x = self.make_tensor((2, 6, 4))
        vec = get_vit_patch_tokens_vectorizer(n_prefixes=2, n_random=None)

        out = vec.vectorize(x)
        assert out.vectors.shape[0] == x.shape[0] * 4
        assert out.vectors.shape[1] == x.shape[2]

    def test_get_vit_patch_tokens_vectorizer_only_random(self):
        x = self.make_tensor((2, 6, 4))
        vec = get_vit_patch_tokens_vectorizer(n_prefixes=None, n_random=2)

        out = vec.vectorize(x)
        assert out.vectors.shape[0] == x.shape[0] * 2
        assert out.vectors.shape[1] == x.shape[2]

    def test_get_vit_patch_tokens_vectorizer_with_nothing(self):
        x = self.make_tensor((2, 6, 4))
        vec = get_vit_patch_tokens_vectorizer(n_prefixes=None, n_random=None)

        out = vec.vectorize(x)
        assert out.vectors.shape[0] == x.shape[0] * 6
        assert out.vectors.shape[1] == x.shape[2]

    def test_get_vit_patch_tokens_vectorizer_remove_and_random(self):
        x = self.make_tensor((2, 8, 4))
        generator = torch.Generator().manual_seed(0)
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
        assert out.vectors.shape[0] == x.shape[0] * 3
        assert out.vectors.shape[1] == x.shape[2]
