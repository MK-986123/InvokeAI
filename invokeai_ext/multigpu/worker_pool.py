"""Worker pool orchestrator for the multi-GPU milestone."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem

from .gpu_worker import GPUWorker, WorkerResult, WorkerTask
from .scheduler import VRAMAwareScheduler


@dataclass(slots=True)
class WorkerAssignment:
    """Tracks a queue item's execution on a worker."""

    item_id: int
    device_id: int
    cancel_event: threading.Event


class WorkerPool:
    """Manage GPU workers and their lifecycle with VRAM-aware scheduling."""

    def __init__(
        self,
        services: InvocationServices,
        runner_template,
        profiler,
    ) -> None:
        self._services = services
        self._runner_template = runner_template
        self._profiler = profiler
        self._scheduler = VRAMAwareScheduler(logger=services.logger)
        self._result_queue: "queue.Queue[WorkerResult]" = queue.Queue()
        self._task_queues: Dict[int, "queue.Queue[Optional[WorkerTask]]"] = {}
        self._workers: Dict[int, GPUWorker] = {}
        self._assignments: Dict[int, WorkerAssignment] = {}
        self._availability_condition = threading.Condition()
        self._available_devices: set[int] = set()
        self._stopping = False

    @property
    def active(self) -> bool:
        return bool(self._workers)

    def _runner_factory(self):
        return self._runner_template.clone()

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        device_ids = self._scheduler.devices
        if len(device_ids) <= 1:
            return
        for device_id in device_ids:
            task_queue: "queue.Queue[Optional[WorkerTask]]" = queue.Queue()
            worker = GPUWorker(
                device_id=device_id,
                services=self._services,
                task_queue=task_queue,
                result_queue=self._result_queue,
                runner_factory=self._runner_factory,
                ready_callback=self._notify_available,
                profiler=self._profiler,
                logger=self._services.logger,
            )
            worker.start()
            self._task_queues[device_id] = task_queue
            self._workers[device_id] = worker

    def submit(self, queue_item: SessionQueueItem) -> Optional[WorkerAssignment]:
        if not self.active:
            return None
        with self._availability_condition:
            while not self._available_devices and not self._stopping:
                self._availability_condition.wait()
            if self._stopping:
                return None
            device_id = self._scheduler.select_device(self._available_devices)
            self._available_devices.discard(device_id)
        cancel_event = threading.Event()
        assignment = WorkerAssignment(
            item_id=queue_item.item_id,
            device_id=device_id,
            cancel_event=cancel_event,
        )
        self._assignments[queue_item.item_id] = assignment
        self._task_queues[device_id].put(WorkerTask(queue_item=queue_item, cancel_event=cancel_event))
        return assignment

    def next_result(self, timeout: Optional[float] = None) -> Optional[WorkerResult]:
        if not self.active:
            return None
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _notify_available(self, device_id: int) -> None:
        with self._availability_condition:
            if self._stopping:
                return
            self._available_devices.add(device_id)
            self._availability_condition.notify_all()

    def cancel(self, item_id: int) -> None:
        assignment = self._assignments.get(item_id)
        if assignment:
            assignment.cancel_event.set()

    def complete(self, item_id: int) -> None:
        self._assignments.pop(item_id, None)

    def cancel_all(self) -> None:
        for assignment in list(self._assignments.values()):
            assignment.cancel_event.set()

    def active_item_ids(self) -> set[int]:
        return set(self._assignments.keys())

    def stop(self) -> None:
        self._stopping = True
        with self._availability_condition:
            self._available_devices.clear()
            self._availability_condition.notify_all()
        for worker in self._workers.values():
            worker.shutdown()
        for worker in self._workers.values():
            worker.join(timeout=5)
        self._workers.clear()
        self._task_queues.clear()
        self._assignments.clear()
        self._scheduler.shutdown()


def start_worker_pool(services: InvocationServices, session_runner, profiler) -> Optional[WorkerPool]:
    if not hasattr(session_runner, "clone"):
        services.logger.warning("Session runner does not support cloning; skipping multi-GPU worker pool startup")
        return None
    pool = WorkerPool(services=services, runner_template=session_runner, profiler=profiler)
    pool.start()
    if pool.active:
        return pool
    pool.stop()
    return None


def stop_worker_pool(pool: Optional[WorkerPool]) -> None:
    if pool is not None:
        pool.stop()
