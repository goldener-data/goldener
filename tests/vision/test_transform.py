import pytest
import torch

from goldener.vision.transform import PatchifyImageMask


class TestPatchifyImageMask:
    def test_output_shape_and_dtype(self):
        mask = torch.zeros(2, 1, 32, 32)
        mask[:, :, 0:16, 0:16] = 1.0

        transform = PatchifyImageMask(patch_size=16, match_ratio=0.5)
        out = transform.transform(mask)

        assert out.shape == (2, 4)
        assert out.dtype == torch.float32
        assert torch.all((out == 0) | (out == 1))

    def test_invalid_ndim_raises(self):
        transform = PatchifyImageMask(patch_size=16, match_ratio=0.5)
        mask = torch.zeros(32, 32)
        with pytest.raises(ValueError, match="Input tensor must have shape"):
            transform.transform(mask)

    def test_non_divisible_size_raises(self):
        transform = PatchifyImageMask(patch_size=16, match_ratio=0.5)
        mask = torch.zeros(1, 1, 17, 32)
        with pytest.raises(ValueError, match="H and W must be divisible by patch_size"):
            transform.transform(mask)

    def test_patchify_logic_and_match_ratio(self):
        mask = torch.zeros(1, 1, 4, 4)

        # top-left patch: all zeros -> ratio 0.0
        # top-right patch: one "1" -> ratio 1/4 = 0.25
        mask[0, 0, 0, 2] = 1.0
        # bottom-left patch: two "1" -> ratio 2/4 = 0.5
        mask[0, 0, 2, 0] = 1.0
        mask[0, 0, 3, 1] = 1.0
        # bottom-right patch: three "1" -> ratio 3/4 = 0.75
        mask[0, 0, 2, 2] = 1.0
        mask[0, 0, 2, 3] = 1.0
        mask[0, 0, 3, 2] = 1.0

        # With match_ratio=0.5 and strict > comparison, expected ratios > 0.5 are only 0.75
        transform = PatchifyImageMask(patch_size=2, match_ratio=0.5)
        out = transform.transform(mask)
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        assert torch.equal(out, expected)

        # With a lower threshold, more patches should pass
        transform_low = PatchifyImageMask(patch_size=2, match_ratio=0.2)
        out_low = transform_low.transform(mask)
        expected_low = torch.tensor([[0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        assert torch.equal(out_low, expected_low)
