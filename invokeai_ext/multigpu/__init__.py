"""Multi-GPU milestone helpers with PCIe safeguards."""
from .gpu_worker import GPUWorker, WorkerResult, WorkerTask
from .model_cache import MultiDeviceModelCache
from .scheduler import VRAMAwareScheduler
from .worker_pool import WorkerAssignment, WorkerPool, start_worker_pool, stop_worker_pool

__all__ = [
    "GPUWorker",
    "WorkerResult",
    "WorkerTask",
    "WorkerAssignment",
    "WorkerPool",
    "VRAMAwareScheduler",
    "MultiDeviceModelCache",
    "start_worker_pool",
    "stop_worker_pool",
]
