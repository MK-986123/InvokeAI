"""VRAM-aware scheduling utilities for the multi-GPU milestone."""
from __future__ import annotations

from typing import Iterable, List

import torch

try:  # pragma: no cover - optional dependency
    import pynvml
except Exception:  # noqa: BLE001
    pynvml = None


class VRAMAwareScheduler:
    """Choose CUDA devices for new sessions while guarding against PCIe stalls."""

    def __init__(self, logger) -> None:
        self._logger = logger
        self._device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._round_robin_index = 0
        self._handles = {}
        self._nvml_ready = False
        if pynvml is None or self._device_count == 0:
            return
        try:
            pynvml.nvmlInit()
            for device_id in range(self._device_count):
                try:
                    self._handles[device_id] = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                except pynvml.NVMLError as err:  # type: ignore[attr-defined]
                    self._logger.warning(
                        "NVML failed to acquire handle for device %s: %s", device_id, err
                    )
            if self._handles:
                self._nvml_ready = True
        except Exception as err:  # noqa: BLE001
            self._logger.warning("Unable to initialize NVML, falling back to round-robin: %s", err)

    def shutdown(self) -> None:
        if self._nvml_ready:
            try:  # pragma: no cover - defensive cleanup
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._nvml_ready = False

    @property
    def devices(self) -> List[int]:
        """Return the CUDA devices managed by this scheduler."""

        return list(range(self._device_count))

    def select_device(self, available: Iterable[int]) -> int:
        """Select the best device from ``available`` using NVML or round-robin."""

        available_list = list(available)
        if not available_list:
            raise RuntimeError("No CUDA devices are available for scheduling")

        if not self._nvml_ready:
            index = self._round_robin_index % len(available_list)
            self._round_robin_index += 1
            return available_list[index]

        best_device = available_list[0]
        best_free = -1
        for device_id in available_list:
            handle = self._handles.get(device_id)
            if handle is None:
                continue
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mem = info.free
            except pynvml.NVMLError as err:  # type: ignore[attr-defined]
                self._logger.warning(
                    "NVML memory check failed for device %s: %s", device_id, err
                )
                free_mem = -1
            if free_mem > best_free:
                best_free = free_mem
                best_device = device_id

        if best_free < 0:
            index = self._round_robin_index % len(available_list)
            self._round_robin_index += 1
            return available_list[index]
        return best_device
