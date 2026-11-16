"""Small synchronous adapters for the existing SGLang-Omni stream relay.

The H3 scheduler stage is synchronous while Omni's relay is async.  Keep the
adapter deliberately narrow: a background loop owns one ShmRelay, and the
control plane carries only a DataRef dictionary plus a completion id.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any

import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.relay.shm import ShmRelay


class _AsyncLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="h3-shm-relay", daemon=True)
        self.thread.start()
        self.ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def run(self, coroutine) -> Any:
        future: Future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.loop.close()


class H3ShmOutputWriter:
    """Publish CPU RGB through Omni's SHM stream relay and retain ACK state."""

    def __init__(self, *, engine_id: str = "h3-rgb-output", credits: int = 2) -> None:
        self._async = _AsyncLoop()
        self._relay = ShmRelay(engine_id=engine_id, credits=credits, device="cpu")
        self._pending: dict[str, tuple[object, ...]] = {}

    def put(self, tensor: torch.Tensor, *, request_id: str, chunk_id: int) -> dict[str, object]:
        if tensor.device.type != "cpu" or tensor.dtype != torch.uint8:
            raise ValueError("H3 SHM RGB output requires a CPU uint8 tensor")
        data_ref, operations = self._async.run(
            stage_io.write_stream_chunk(
                self._relay,
                request_id=request_id,
                data=tensor,
                target_stage="h3_bridge",
                from_stage="h3_scheduler",
                chunk_id=chunk_id,
                metadata={"frame_count": int(tensor.shape[0])},
                transport=TransportKind.SHM,
            )
        )
        self._pending[data_ref.object_id] = tuple(operations)
        return {"_type": "H3OmniStreamRef", "data_ref": data_ref.to_dict()}

    async def _ack_async(self, operations: tuple[object, ...]) -> None:
        for operation in operations:
            operation.mark_receiver_done()
            await operation.wait_for_completion()

    def ack(self, object_ids: tuple[str, ...]) -> None:
        for object_id in object_ids:
            operations = self._pending.pop(object_id, None)
            if operations is not None:
                self._async.run(self._ack_async(operations))

    def close(self) -> None:
        self._async.close()


class H3ShmOutputReader:
    """Materialize an Omni stream DataRef on the bridge process."""

    def __init__(self, *, engine_id: str = "h3-rgb-reader", credits: int = 2) -> None:
        self._async = _AsyncLoop()
        self._relay = ShmRelay(engine_id=engine_id, credits=credits, device="cpu")

    def read(self, value: dict[str, object]):
        if value.get("_type") != "H3OmniStreamRef":
            raise ValueError("unexpected H3 RGB transport reference")
        data_ref = DataRef.from_dict(value["data_ref"])
        tensor, metadata = self._async.run(
            stage_io.read_stream_chunk(self._relay, data_ref)
        )
        return tensor.numpy(), data_ref.object_id, metadata

    def close(self) -> None:
        self._async.close()
