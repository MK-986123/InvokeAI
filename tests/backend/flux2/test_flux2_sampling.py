# Copyright (c) 2024, InvokeAI Development Team
"""Tests for FLUX.2 sampling utilities."""

import math

import pytest
import torch

from invokeai.backend.flux2.sampling_utils import (
    clip_flux2_timestep_schedule,
    exponential_time_shift,
    generate_flux2_img_ids,
    generate_flux2_txt_ids,
    get_flux2_noise,
    get_flux2_schedule,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.flux2.util import FLUX2_LATENT_CHANNELS


class TestExponentialTimeShift:
    """Tests for exponential time-shifting function."""

    def test_shift_zero_returns_original(self):
        """Test that shift=0 returns the original values."""
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        t_shifted = exponential_time_shift(t, shift=0.0)
        assert torch.allclose(t, t_shifted)

    def test_shift_preserves_endpoints(self):
        """Test that 0 and 1 map to 0 and 1 respectively."""
        t = torch.tensor([0.0, 1.0])
        t_shifted = exponential_time_shift(t, shift=3.0)
        assert t_shifted[0].item() == pytest.approx(0.0, abs=1e-6)
        assert t_shifted[1].item() == pytest.approx(1.0, abs=1e-6)

    def test_shift_moves_values_toward_one(self):
        """Test that positive shift biases values toward 1 (high timesteps)."""
        t = torch.tensor([0.5])
        t_shifted = exponential_time_shift(t, shift=3.0)
        # With positive shift, 0.5 should become > 0.5
        assert t_shifted[0].item() > 0.5

    def test_shift_formula_correctness(self):
        """Test the exponential shift formula: t_shifted = (e^(μ*t) - 1) / (e^μ - 1)."""
        t = torch.tensor([0.5])
        shift = 3.0

        t_shifted = exponential_time_shift(t, shift=shift)

        # Manual calculation
        expected = (math.exp(shift * 0.5) - 1.0) / (math.exp(shift) - 1.0)
        assert t_shifted[0].item() == pytest.approx(expected, abs=1e-6)


class TestFlux2Schedule:
    """Tests for FLUX.2 timestep schedule generation."""

    def test_schedule_length(self):
        """Test that schedule has num_steps + 1 entries."""
        schedule = get_flux2_schedule(num_steps=4, shift=3.0)
        assert len(schedule) == 5  # 4 steps + 1 for terminal value

    def test_schedule_endpoints(self):
        """Test that schedule starts at 1.0 and ends at 0.0."""
        schedule = get_flux2_schedule(num_steps=4, shift=3.0)
        assert schedule[0] == pytest.approx(1.0, abs=1e-6)
        assert schedule[-1] == pytest.approx(0.0, abs=1e-6)

    def test_schedule_monotonically_decreasing(self):
        """Test that schedule is monotonically decreasing."""
        schedule = get_flux2_schedule(num_steps=10, shift=3.0)
        for i in range(len(schedule) - 1):
            assert schedule[i] > schedule[i + 1]

    def test_schedule_no_shift(self):
        """Test linear schedule when exponential shift is disabled."""
        schedule = get_flux2_schedule(num_steps=4, shift=0.0, use_exponential_shift=False)
        expected = [1.0, 0.75, 0.5, 0.25, 0.0]
        for s, e in zip(schedule, expected):
            assert s == pytest.approx(e, abs=1e-6)

    def test_schedule_with_shift_biased(self):
        """Test that shifted schedule is biased toward high timesteps."""
        schedule_no_shift = get_flux2_schedule(num_steps=4, use_exponential_shift=False)
        schedule_with_shift = get_flux2_schedule(num_steps=4, shift=3.0, use_exponential_shift=True)

        # Middle values should be higher with shift (biased toward 1)
        assert schedule_with_shift[2] > schedule_no_shift[2]


class TestFlux2ClipSchedule:
    """Tests for timestep schedule clipping."""

    def test_clip_no_change(self):
        """Test that full range doesn't change schedule."""
        schedule = get_flux2_schedule(num_steps=4, shift=3.0)
        clipped = clip_flux2_timestep_schedule(schedule, denoising_start=0.0, denoising_end=1.0)
        assert len(clipped) == len(schedule)

    def test_clip_start_only(self):
        """Test clipping with denoising_start > 0."""
        schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        clipped = clip_flux2_timestep_schedule(schedule, denoising_start=0.2, denoising_end=1.0)
        # Should start at t=0.8 (1.0 - 0.2)
        assert clipped[0] == pytest.approx(0.8, abs=1e-6)

    def test_clip_end_only(self):
        """Test clipping with denoising_end < 1."""
        schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        clipped = clip_flux2_timestep_schedule(schedule, denoising_start=0.0, denoising_end=0.8)
        # Should end at t=0.2 (1.0 - 0.8)
        assert clipped[-1] == pytest.approx(0.2, abs=1e-6)


class TestFlux2Noise:
    """Tests for FLUX.2 noise generation."""

    def test_noise_shape(self):
        """Test that noise has correct shape for FLUX.2."""
        noise = get_flux2_noise(
            num_samples=1,
            height=1024,
            width=1024,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=42,
        )
        # FLUX.2: 32 channels, 8x compression, rounded to even
        expected_shape = (1, FLUX2_LATENT_CHANNELS, 128, 128)
        assert noise.shape == expected_shape

    def test_noise_channels(self):
        """Test that noise has 32 channels for FLUX.2."""
        noise = get_flux2_noise(
            num_samples=1,
            height=512,
            width=512,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=0,
        )
        assert noise.shape[1] == 32  # FLUX.2 uses 32 latent channels

    def test_noise_deterministic(self):
        """Test that same seed produces same noise."""
        noise1 = get_flux2_noise(1, 256, 256, torch.device("cpu"), torch.float32, seed=123)
        noise2 = get_flux2_noise(1, 256, 256, torch.device("cpu"), torch.float32, seed=123)
        assert torch.allclose(noise1, noise2)

    def test_noise_different_seeds(self):
        """Test that different seeds produce different noise."""
        noise1 = get_flux2_noise(1, 256, 256, torch.device("cpu"), torch.float32, seed=123)
        noise2 = get_flux2_noise(1, 256, 256, torch.device("cpu"), torch.float32, seed=456)
        assert not torch.allclose(noise1, noise2)


class TestFlux2PackUnpack:
    """Tests for FLUX.2 latent packing/unpacking."""

    def test_pack_shape(self):
        """Test that pack_flux2 produces correct output shape."""
        # Input: [B, 32, H, W]
        x = torch.randn(1, 32, 64, 64)
        packed = pack_flux2(x)
        # Output should be [B, (H/2)*(W/2), 32*2*2=128]
        assert packed.shape == (1, 32 * 32, 128)

    def test_unpack_shape(self):
        """Test that unpack_flux2 produces correct output shape."""
        # Packed: [B, L, 128] where L = (H/16) * (W/16)
        packed = torch.randn(1, 32 * 32, 128)
        unpacked = unpack_flux2(packed, height=512, width=512)
        # Should restore to [B, 32, H/8, W/8]
        assert unpacked.shape == (1, 32, 64, 64)

    def test_pack_unpack_roundtrip(self):
        """Test that pack -> unpack is identity."""
        x = torch.randn(1, 32, 64, 64)
        packed = pack_flux2(x)
        unpacked = unpack_flux2(packed, height=512, width=512)
        assert torch.allclose(x, unpacked)


class TestFlux2PositionIds:
    """Tests for FLUX.2 position ID generation."""

    def test_img_ids_shape(self):
        """Test image position IDs shape."""
        img_ids = generate_flux2_img_ids(
            h=64, w=64, batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        # After packing: (H/2) * (W/2) positions, 4 axes for FLUX.2
        expected_shape = (1, 32 * 32, 4)
        assert img_ids.shape == expected_shape

    def test_img_ids_axes(self):
        """Test that image position IDs have correct axis values."""
        img_ids = generate_flux2_img_ids(
            h=8, w=8, batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        # Axis 0: batch offset (all 0)
        assert (img_ids[0, :, 0] == 0).all()
        # Axis 3: extra dimension (all 0)
        assert (img_ids[0, :, 3] == 0).all()

    def test_txt_ids_shape(self):
        """Test text position IDs shape."""
        txt_ids = generate_flux2_txt_ids(
            seq_len=512, batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        expected_shape = (1, 512, 4)
        assert txt_ids.shape == expected_shape
