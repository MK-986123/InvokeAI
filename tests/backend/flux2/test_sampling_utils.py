"""Tests for FLUX.2 sampling utilities."""

import math

import torch

from invokeai.backend.flux2.sampling_utils import (
    compute_empirical_mu,
    generate_img_ids_flux2,
    get_noise_flux2,
    get_schedule_flux2,
    pack_flux2,
    unpack_flux2,
)


class TestGetNoiseFlux2:
    """Tests for 32-channel noise generation."""

    def test_noise_shape(self):
        """Test that noise has correct 32-channel shape."""
        noise = get_noise_flux2(
            num_samples=1, height=1024, width=1024, device=torch.device("cpu"), dtype=torch.float32, seed=42
        )
        # height/8 = 128, but then must be divisible by 2: 2*ceil(1024/16) = 128
        assert noise.shape == (1, 32, 128, 128)

    def test_noise_shape_non_square(self):
        """Test noise shape for non-square images."""
        noise = get_noise_flux2(
            num_samples=1, height=768, width=512, device=torch.device("cpu"), dtype=torch.float32, seed=42
        )
        latent_h = 2 * math.ceil(768 / 16)  # 96
        latent_w = 2 * math.ceil(512 / 16)  # 64
        assert noise.shape == (1, 32, latent_h, latent_w)

    def test_noise_reproducibility(self):
        """Test that same seed produces same noise."""
        noise1 = get_noise_flux2(
            num_samples=1, height=512, width=512, device=torch.device("cpu"), dtype=torch.float32, seed=123
        )
        noise2 = get_noise_flux2(
            num_samples=1, height=512, width=512, device=torch.device("cpu"), dtype=torch.float32, seed=123
        )
        assert torch.allclose(noise1, noise2)


class TestPackUnpackFlux2:
    """Tests for packing and unpacking FLUX.2 latents."""

    def test_pack_shape(self):
        """Test that packing produces correct sequence shape."""
        x = torch.randn(1, 32, 128, 128)
        packed = pack_flux2(x)
        # (B, 32, 128, 128) -> (B, 64*64, 128) = (1, 4096, 128)
        assert packed.shape == (1, 64 * 64, 32 * 2 * 2)
        assert packed.shape == (1, 4096, 128)

    def test_unpack_shape(self):
        """Test that unpacking produces correct spatial shape."""
        packed = torch.randn(1, 4096, 128)
        unpacked = unpack_flux2(packed, height=1024, width=1024)
        assert unpacked.shape == (1, 32, 128, 128)

    def test_pack_unpack_roundtrip(self):
        """Test that pack->unpack is identity."""
        x = torch.randn(1, 32, 64, 64)
        packed = pack_flux2(x)
        unpacked = unpack_flux2(packed, height=512, width=512)
        assert torch.allclose(x, unpacked)


class TestGenerateImgIdsFlux2:
    """Tests for 4D position ID generation."""

    def test_img_ids_shape(self):
        """Test that img_ids have correct shape."""
        img_ids = generate_img_ids_flux2(h=128, w=128, batch_size=1, device=torch.device("cpu"))
        packed_h = 128 // 2  # 64
        packed_w = 128 // 2  # 64
        assert img_ids.shape == (1, packed_h * packed_w, 4)

    def test_img_ids_dtype(self):
        """Test that img_ids use int64 dtype (required for RoPE)."""
        img_ids = generate_img_ids_flux2(h=64, w=64, batch_size=1, device=torch.device("cpu"))
        assert img_ids.dtype == torch.long

    def test_img_ids_first_two_dims_zero(self):
        """Test that T and L dims are zero (only H, W vary)."""
        img_ids = generate_img_ids_flux2(h=64, w=64, batch_size=1, device=torch.device("cpu"))
        # First dim (T) should be all zeros
        assert (img_ids[..., 0] == 0).all()
        # Fourth dim (L) should be all zeros
        assert (img_ids[..., 3] == 0).all()

    def test_img_ids_h_w_values(self):
        """Test that H and W position values are correct."""
        img_ids = generate_img_ids_flux2(h=8, w=8, batch_size=1, device=torch.device("cpu"))
        packed_h = 4
        packed_w = 4
        # H values should range from 0 to packed_h-1
        h_values = img_ids[0, :, 1].reshape(packed_h, packed_w)
        assert (h_values[:, 0] == torch.arange(packed_h)).all()
        # W values should range from 0 to packed_w-1
        w_values = img_ids[0, :, 2].reshape(packed_h, packed_w)
        assert (w_values[0, :] == torch.arange(packed_w)).all()


class TestComputeEmpiricalMu:
    """Tests for empirical mu computation."""

    def test_mu_positive(self):
        """Test that mu is positive for typical values."""
        mu = compute_empirical_mu(image_seq_len=4096, num_steps=4)
        assert mu > 0

    def test_mu_large_image(self):
        """Test mu for large image (>4300 tokens)."""
        mu = compute_empirical_mu(image_seq_len=5000, num_steps=28)
        assert mu > 0

    def test_mu_increases_with_seq_len(self):
        """Test that mu increases with image sequence length."""
        mu1 = compute_empirical_mu(image_seq_len=1024, num_steps=20)
        mu2 = compute_empirical_mu(image_seq_len=4096, num_steps=20)
        assert mu2 > mu1


class TestGetScheduleFlux2:
    """Tests for FLUX.2 sigma schedule generation."""

    def test_schedule_length(self):
        """Test that schedule has num_steps + 1 entries."""
        schedule = get_schedule_flux2(num_steps=4, image_seq_len=4096)
        assert len(schedule) == 5  # 4 steps + final 0.0

    def test_schedule_starts_at_one(self):
        """Test that schedule starts at 1.0."""
        schedule = get_schedule_flux2(num_steps=4, image_seq_len=4096)
        assert schedule[0] == 1.0

    def test_schedule_ends_at_zero(self):
        """Test that schedule ends at 0.0."""
        schedule = get_schedule_flux2(num_steps=4, image_seq_len=4096)
        assert schedule[-1] == 0.0

    def test_schedule_is_descending(self):
        """Test that schedule values are monotonically decreasing."""
        schedule = get_schedule_flux2(num_steps=10, image_seq_len=4096)
        for i in range(len(schedule) - 1):
            assert schedule[i] > schedule[i + 1]
