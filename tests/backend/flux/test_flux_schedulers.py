"""Tests for FLUX scheduler functionality."""

import torch
import pytest

from invokeai.backend.flux.flux_schedulers import (
    FLUX_SCHEDULER_MAP,
    FLUX_SCHEDULER_NAME,
    FLUX_SCHEDULER_PARAMS,
)


def test_flux_scheduler_map_keys():
    """Test that all scheduler names are defined in both map and params."""
    assert set(FLUX_SCHEDULER_MAP.keys()) == set(FLUX_SCHEDULER_PARAMS.keys())


@pytest.mark.parametrize("scheduler_name", FLUX_SCHEDULER_MAP.keys())
def test_scheduler_instantiation(scheduler_name: FLUX_SCHEDULER_NAME):
    """Test that each scheduler can be instantiated with its parameters."""
    scheduler_cls = FLUX_SCHEDULER_MAP[scheduler_name]
    scheduler_params = FLUX_SCHEDULER_PARAMS[scheduler_name]
    
    # Instantiate the scheduler
    scheduler = scheduler_cls(num_train_timesteps=1000, **scheduler_params)
    
    # Verify it has the expected interface
    assert hasattr(scheduler, "set_timesteps")
    assert hasattr(scheduler, "timesteps")


@pytest.mark.parametrize("scheduler_name", FLUX_SCHEDULER_MAP.keys())
def test_scheduler_timesteps(scheduler_name: FLUX_SCHEDULER_NAME):
    """Test that schedulers produce monotonic timesteps that can be normalized to [0,1]."""
    scheduler_cls = FLUX_SCHEDULER_MAP[scheduler_name]
    scheduler_params = FLUX_SCHEDULER_PARAMS[scheduler_name]
    
    scheduler = scheduler_cls(num_train_timesteps=1000, **scheduler_params)
    scheduler.set_timesteps(num_inference_steps=20, device=torch.device("cpu"))
    
    raw_timesteps = scheduler.timesteps.float().cpu().tolist()
    # Normalize to [0, 1] format like FLUX expects
    timesteps = [t / 1000.0 for t in raw_timesteps]
    
    # Verify timesteps are between 0 and 1 (approximately)
    assert all(0.0 <= t <= 1.0 for t in timesteps), f"Normalized timesteps out of range for {scheduler_name}: {timesteps}"
    
    # Verify timesteps are monotonic decreasing (flow matching goes from 1 to 0)
    assert timesteps == sorted(timesteps, reverse=True), f"Timesteps not monotonic for {scheduler_name}: {timesteps}"
    
    # Verify we have timesteps (Heun may produce more due to its method)
    assert len(timesteps) > 0, f"No timesteps produced for {scheduler_name}"


def test_karras_vs_regular_euler():
    """Test that Karras and regular Euler schedulers produce different timesteps."""
    regular_scheduler = FLUX_SCHEDULER_MAP["flow_euler"](
        num_train_timesteps=1000, **FLUX_SCHEDULER_PARAMS["flow_euler"]
    )
    karras_scheduler = FLUX_SCHEDULER_MAP["flow_euler_k"](
        num_train_timesteps=1000, **FLUX_SCHEDULER_PARAMS["flow_euler_k"]
    )
    
    regular_scheduler.set_timesteps(num_inference_steps=10, device=torch.device("cpu"))
    karras_scheduler.set_timesteps(num_inference_steps=10, device=torch.device("cpu"))
    
    regular_timesteps = [t / 1000.0 for t in regular_scheduler.timesteps.float().cpu().tolist()]
    karras_timesteps = [t / 1000.0 for t in karras_scheduler.timesteps.float().cpu().tolist()]
    
    # They should be different
    assert regular_timesteps != karras_timesteps, "Karras and regular Euler should produce different timesteps"


def test_exponential_vs_regular_euler():
    """Test that exponential and regular Euler schedulers produce different timesteps."""
    regular_scheduler = FLUX_SCHEDULER_MAP["flow_euler"](
        num_train_timesteps=1000, **FLUX_SCHEDULER_PARAMS["flow_euler"]
    )
    exp_scheduler = FLUX_SCHEDULER_MAP["flow_euler_exp"](
        num_train_timesteps=1000, **FLUX_SCHEDULER_PARAMS["flow_euler_exp"]
    )
    
    regular_scheduler.set_timesteps(num_inference_steps=10, device=torch.device("cpu"))
    exp_scheduler.set_timesteps(num_inference_steps=10, device=torch.device("cpu"))
    
    regular_timesteps = [t / 1000.0 for t in regular_scheduler.timesteps.float().cpu().tolist()]
    exp_timesteps = [t / 1000.0 for t in exp_scheduler.timesteps.float().cpu().tolist()]
    
    # They should be different
    assert regular_timesteps != exp_timesteps, "Exponential and regular Euler should produce different timesteps"