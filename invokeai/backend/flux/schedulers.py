"""Flow Matching scheduler definitions and mapping.

This module provides the scheduler types and mapping for Flow Matching models
(Flux, FLUX.2, and Z-Image), supporting multiple schedulers from the diffusers library.
"""

from typing import Literal, Type

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin

# Note: FlowMatchLCMScheduler may not be available in all diffusers versions
try:
    from diffusers import FlowMatchLCMScheduler

    _HAS_LCM = True
except ImportError:
    _HAS_LCM = False

# Scheduler name literal type for type checking
FLUX_SCHEDULER_NAME_VALUES = Literal["euler", "heun", "lcm"]

# Human-readable labels for the UI
FLUX_SCHEDULER_LABELS: dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun (2nd order)",
    "lcm": "LCM",
}

# Mapping from scheduler names to scheduler classes
FLUX_SCHEDULER_MAP: dict[str, Type[SchedulerMixin]] = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": FlowMatchHeunDiscreteScheduler,
}

if _HAS_LCM:
    FLUX_SCHEDULER_MAP["lcm"] = FlowMatchLCMScheduler


# FLUX.2 scheduler types
# FLUX.2-klein uses Rectified Flow with exponential time-shifting (shift=3.0)
# Distilled models use 4 steps, base models use 20-50 steps
FLUX2_SCHEDULER_NAME_VALUES = Literal["euler", "heun"]

FLUX2_SCHEDULER_LABELS: dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun (2nd order)",
}

FLUX2_SCHEDULER_MAP: dict[str, Type[SchedulerMixin]] = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": FlowMatchHeunDiscreteScheduler,
}


# Z-Image scheduler types (same schedulers as Flux, both use Flow Matching)
# Note: Z-Image-Turbo is optimized for ~8 steps with Euler, but other schedulers
# can be used for experimentation.
ZIMAGE_SCHEDULER_NAME_VALUES = Literal["euler", "heun", "lcm"]

# Human-readable labels for the UI
ZIMAGE_SCHEDULER_LABELS: dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun (2nd order)",
    "lcm": "LCM",
}

# Mapping from scheduler names to scheduler classes (same as Flux)
ZIMAGE_SCHEDULER_MAP: dict[str, Type[SchedulerMixin]] = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": FlowMatchHeunDiscreteScheduler,
}

if _HAS_LCM:
    ZIMAGE_SCHEDULER_MAP["lcm"] = FlowMatchLCMScheduler
