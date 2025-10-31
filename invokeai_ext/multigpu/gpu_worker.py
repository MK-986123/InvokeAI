"""Multi-GPU milestone workers."""
from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem


@dataclass(slots=True)
class WorkerTask:
    """Instruction bundle dispatched to a GPU worker thread."""

    queue_item: SessionQueueItem
    cancel_event: threading.Event


@dataclass(slots=True)
class WorkerResult:
    """Result payload returned by a GPU worker after processing a session."""

    queue_item: Optional[SessionQueueItem]
    device_id: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


class GPUWorker(threading.Thread):
    """Thread that executes Invoke sessions on a dedicated CUDA device."""

    def __init__(
        self,
        *,
        device_id: int,
        services: InvocationServices,
        task_queue: "queue.Queue[Optional[WorkerTask]]",
        result_queue: "queue.Queue[WorkerResult]",
        runner_factory: Callable[[], DefaultSessionRunner],
        ready_callback: Callable[[int], None],
        profiler,
        logger,
    ) -> None:
        super().__init__(name=f"gpu-worker-{device_id}", daemon=True)
        self._device_id = device_id
        self._services = services
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._runner_factory = runner_factory
        self._ready_callback = ready_callback
        self._profiler = profiler
        self._logger = logger
        self._shutdown = threading.Event()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._task_queue.put(None)

    def run(self) -> None:  # pragma: no cover - exercised in integration tests
        try:
            torch.cuda.set_device(self._device_id)
        except Exception as e:  # noqa: BLE001
            self._logger.error(
                "Failed to bind GPU worker to CUDA device %s: %s", self._device_id, e
            )
            traceback_str = traceback.format_exc()
            self._result_queue.put(
                WorkerResult(
                    queue_item=None,
                    device_id=self._device_id,
                    error_type=e.__class__.__name__,
                    error_message=str(e),
                    error_traceback=traceback_str,
                )
            )
            return

        runner = self._runner_factory()

        while not self._shutdown.is_set():
            self._ready_callback(self._device_id)
            task = self._task_queue.get()
            if task is None:
                break

            try:
                runner.start(
                    services=self._services,
                    cancel_event=task.cancel_event,
                    profiler=self._profiler,
                )
                runner.run(queue_item=task.queue_item)
                self._result_queue.put(
                    WorkerResult(queue_item=task.queue_item, device_id=self._device_id)
                )
            except Exception as e:  # noqa: BLE001
                self._logger.error(
                    "Unhandled exception in GPU worker for device %s: %s",
                    self._device_id,
                    e,
                )
                self._result_queue.put(
                    WorkerResult(
                        queue_item=task.queue_item,
                        device_id=self._device_id,
                        error_type=e.__class__.__name__,
                        error_message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )
            finally:
                self._task_queue.task_done()

        # Drain any pending sentinel entries to unblock shutdown
        while True:
            try:
                pending = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if pending is not None:
                self._task_queue.task_done()
        self._ready_callback(self._device_id)

