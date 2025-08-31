# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""FLUX scheduler definitions for flow-matching schedulers compatible with FLUX models."""

from typing import Literal, Type

from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin

# Literal union of supported FLUX schedulers
FLUX_SCHEDULER_NAME = Literal[
    "flow_euler",        # default FlowMatch Euler
    "flow_euler_k",      # Euler with Karras sigmas
    "flow_euler_exp",    # Euler with exponential sigmas
    "flow_heun",         # FlowMatch Heun
]

# Mapping of scheduler names to their classes
FLUX_SCHEDULER_MAP: dict[FLUX_SCHEDULER_NAME, Type[SchedulerMixin]] = {
    "flow_euler": FlowMatchEulerDiscreteScheduler,
    "flow_euler_k": FlowMatchEulerDiscreteScheduler,
    "flow_euler_exp": FlowMatchEulerDiscreteScheduler,
    "flow_heun": FlowMatchHeunDiscreteScheduler,
}

# Parameters to pass to each scheduler
FLUX_SCHEDULER_PARAMS: dict[FLUX_SCHEDULER_NAME, dict[str, any]] = {
    "flow_euler": {"use_karras_sigmas": False, "use_exponential_sigmas": False},
    "flow_euler_k": {"use_karras_sigmas": True, "use_exponential_sigmas": False},
    "flow_euler_exp": {"use_karras_sigmas": False, "use_exponential_sigmas": True},
    "flow_heun": {},  # Heun scheduler currently exposes fewer options
}