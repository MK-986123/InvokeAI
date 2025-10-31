"""Model cache wrappers for the multi-GPU milestone."""
from __future__ import annotations

import threading
from typing import Callable, Dict, Iterable, List, Optional

import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import CacheStats, ModelCache


class MultiDeviceModelCache:
    """Wrap multiple ``ModelCache`` instances and route operations per CUDA device.

    The wrapper safeguards PCIe saturation by provisioning a dedicated cache per GPU
    while respecting partial-loading toggles, RAM mirroring preferences, and per-device
    VRAM budgets derived from ``max_cache_vram_gb``.
    """

    def __init__(
        self,
        *,
        execution_device_working_mem_gb: float,
        enable_partial_loading: bool,
        keep_ram_copy_of_weights: bool,
        max_ram_cache_size_gb: Optional[float] = None,
        max_vram_cache_size_gb: Optional[float] = None,
        log_memory_usage: bool = False,
        logger=None,
    ) -> None:
        self._lock = threading.RLock()
        self._execution_device_working_mem_gb = execution_device_working_mem_gb
        self._enable_partial_loading = enable_partial_loading
        self._keep_ram_copy_of_weights = keep_ram_copy_of_weights
        self._max_ram_cache_size_gb = max_ram_cache_size_gb
        self._total_vram_limit = max_vram_cache_size_gb
        self._log_memory_usage = log_memory_usage
        self._logger = logger
        self._caches: Dict[int, ModelCache] = {}
        self._callbacks_hit: List[Callable] = []
        self._callbacks_miss: List[Callable] = []
        self._callbacks_cleared: List[Callable] = []
        self._callback_tokens_hit: Dict[Callable, List[Callable[[], None]]] = {}
        self._callback_tokens_miss: Dict[Callable, List[Callable[[], None]]] = {}
        self._callback_tokens_cleared: Dict[Callable, List[Callable[[], None]]] = {}
        self._device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._per_device_vram_limit = (
            (self._total_vram_limit / self._device_count)
            if self._total_vram_limit is not None and self._device_count > 0
            else self._total_vram_limit
        )
        if self._device_count == 0:
            # Fallback to a CPU-only cache
            cpu_cache = ModelCache(
                execution_device_working_mem_gb=self._execution_device_working_mem_gb,
                enable_partial_loading=self._enable_partial_loading,
                keep_ram_copy_of_weights=self._keep_ram_copy_of_weights,
                max_ram_cache_size_gb=self._max_ram_cache_size_gb,
                max_vram_cache_size_gb=self._total_vram_limit,
                execution_device="cpu",
                storage_device="cpu",
                log_memory_usage=self._log_memory_usage,
                logger=self._logger,
            )
            self._caches[-1] = cpu_cache

    def _resolve_device(self) -> int:
        if not torch.cuda.is_available():
            return -1
        try:
            return torch.cuda.current_device()
        except Exception:  # noqa: BLE001
            return 0

    def _get_cache(self, device_id: Optional[int] = None) -> ModelCache:
        with self._lock:
            device = self._resolve_device() if device_id is None else device_id
            if device not in self._caches:
                execution_device = f"cuda:{device}" if device >= 0 else "cpu"
                cache = ModelCache(
                    execution_device_working_mem_gb=self._execution_device_working_mem_gb,
                    enable_partial_loading=self._enable_partial_loading,
                    keep_ram_copy_of_weights=self._keep_ram_copy_of_weights,
                    max_ram_cache_size_gb=self._max_ram_cache_size_gb,
                    max_vram_cache_size_gb=self._per_device_vram_limit,
                    execution_device=execution_device,
                    storage_device="cpu",
                    log_memory_usage=self._log_memory_usage,
                    logger=self._logger,
                )
                for cb in self._callbacks_hit:
                    token = cache.on_cache_hit(cb)
                    self._callback_tokens_hit.setdefault(cb, []).append(token)
                for cb in self._callbacks_miss:
                    token = cache.on_cache_miss(cb)
                    self._callback_tokens_miss.setdefault(cb, []).append(token)
                for cb in self._callbacks_cleared:
                    token = cache.on_cache_models_cleared(cb)
                    self._callback_tokens_cleared.setdefault(cb, []).append(token)
                self._caches[device] = cache
            return self._caches[device]

    def put(self, key: str, model) -> None:
        self._get_cache().put(key, model)

    def get(self, *args, **kwargs):
        return self._get_cache().get(*args, **kwargs)

    def make_room(self, *args, **kwargs) -> None:
        self._get_cache().make_room(*args, **kwargs)

    def on_cache_hit(self, cb: Callable) -> Callable[[], None]:
        self._callbacks_hit.append(cb)
        tokens = [cache.on_cache_hit(cb) for cache in self._caches.values()]
        self._callback_tokens_hit[cb] = tokens

        def unsubscribe() -> None:
            self._callbacks_hit[:] = [c for c in self._callbacks_hit if c is not cb]
            for token in self._callback_tokens_hit.pop(cb, []):
                token()

        return unsubscribe

    def on_cache_miss(self, cb: Callable) -> Callable[[], None]:
        self._callbacks_miss.append(cb)
        tokens = [cache.on_cache_miss(cb) for cache in self._caches.values()]
        self._callback_tokens_miss[cb] = tokens

        def unsubscribe() -> None:
            self._callbacks_miss[:] = [c for c in self._callbacks_miss if c is not cb]
            for token in self._callback_tokens_miss.pop(cb, []):
                token()

        return unsubscribe

    def on_cache_models_cleared(self, cb: Callable) -> Callable[[], None]:
        self._callbacks_cleared.append(cb)
        tokens = [cache.on_cache_models_cleared(cb) for cache in self._caches.values()]
        self._callback_tokens_cleared[cb] = tokens

        def unsubscribe() -> None:
            self._callbacks_cleared[:] = [c for c in self._callbacks_cleared if c is not cb]
            for token in self._callback_tokens_cleared.pop(cb, []):
                token()

        return unsubscribe

    @property
    def stats(self) -> Optional[CacheStats]:  # type: ignore[override]
        snapshots = []
        for cache in self._caches.values():
            stats = cache.stats
            if stats is not None:
                snapshots.append(stats)
        if not snapshots:
            return None
        aggregate = CacheStats()
        for stats in snapshots:
            aggregate.hits += stats.hits
            aggregate.misses += stats.misses
            aggregate.high_watermark = max(aggregate.high_watermark, stats.high_watermark)
            aggregate.in_cache += stats.in_cache
            aggregate.cleared += stats.cleared
            aggregate.cache_size += stats.cache_size
            aggregate.loaded_model_sizes.update(stats.loaded_model_sizes)
        return aggregate

    @stats.setter
    def stats(self, stats: CacheStats) -> None:  # type: ignore[override]
        for cache in self._caches.values():
            cache.stats = stats

    def caches(self) -> Iterable[ModelCache]:
        return list(self._caches.values())
