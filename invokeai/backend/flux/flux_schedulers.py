# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""FLUX scheduler definitions for flow-matching schedulers compatible with FLUX models."""

from typing import Literal, Type

from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin

# Literal union of supported FLUX schedulers
FLUX_SCHEDULER_NAME = Literal[
    "flow_euler",  # default FlowMatch Euler
    "flow_euler_k",  # Euler with Karras sigmas
    "flow_euler_exp",  # Euler with exponential sigmas
    "flow_euler_dyn",  # Euler with dynamic shifting (exponential)
    "flow_euler_dyn_lin",  # Euler with dynamic shifting (linear)
    "flow_euler_max",  # Euler with widened shift window for aggressive cleanup
    "flow_heun",  # FlowMatch Heun
    "flow_heun_shift",  # Heun with stronger shift for detail recovery
    "flow_heun_fast",  # Heun with reduced shift for faster, subtle cleanup
]

# Mapping of scheduler names to their classes
FLUX_SCHEDULER_MAP: dict[FLUX_SCHEDULER_NAME, Type[SchedulerMixin]] = {
    "flow_euler": FlowMatchEulerDiscreteScheduler,
    "flow_euler_k": FlowMatchEulerDiscreteScheduler,
    "flow_euler_exp": FlowMatchEulerDiscreteScheduler,
    "flow_euler_dyn": FlowMatchEulerDiscreteScheduler,
    "flow_euler_dyn_lin": FlowMatchEulerDiscreteScheduler,
    "flow_euler_max": FlowMatchEulerDiscreteScheduler,
    "flow_heun": FlowMatchHeunDiscreteScheduler,
    "flow_heun_shift": FlowMatchHeunDiscreteScheduler,
    "flow_heun_fast": FlowMatchHeunDiscreteScheduler,
}

# Parameters to pass to each scheduler
FLUX_SCHEDULER_PARAMS: dict[FLUX_SCHEDULER_NAME, dict[str, any]] = {
    "flow_euler": {"use_karras_sigmas": False, "use_exponential_sigmas": False},
    "flow_euler_k": {"use_karras_sigmas": True, "use_exponential_sigmas": False},
    "flow_euler_exp": {"use_karras_sigmas": False, "use_exponential_sigmas": True},
    "flow_euler_dyn": {
        "use_dynamic_shifting": True,
        "time_shift_type": "exponential",
    },
    "flow_euler_dyn_lin": {
        "use_dynamic_shifting": True,
        "time_shift_type": "linear",
    },
    "flow_euler_max": {
        "base_shift": 0.35,
        "max_shift": 1.35,
        "use_dynamic_shifting": True,
    },
    "flow_heun": {},  # Heun scheduler currently exposes fewer options
    "flow_heun_shift": {"shift": 1.2},
    "flow_heun_fast": {"shift": 0.85},
}
