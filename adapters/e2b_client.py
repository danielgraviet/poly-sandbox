"""E2B sandbox adapter for PolySandbox."""

from __future__ import annotations
import os
import time
import asyncio
import dataclasses
from typing import List
import logging

from e2b import Sandbox  # official E2B SDK
import adapters.base_client as base_client

logger = logging.getLogger(__name__)

class E2BPool:
    """Manages a pool of persistent E2B sandboxes for concurrent execution."""

    def __init__(self, size: int = 5, template_id: str | None = None):
        self._size = size
        self._sandboxes: List[Sandbox] = []
        self._template_id = template_id or os.getenv("E2B_TEMPLATE_ID")
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[Sandbox] = asyncio.Queue()

    async def start(self):
        """Provision all E2B sandboxes asynchronously."""
        logger.info(f"[E2BPool] Spinning up {self._size} sandboxes...")
        for i in range(self._size):
            sb = await asyncio.to_thread(Sandbox.create, template_id=self._template_id)
            self._sandboxes.append(sb)
            await self._queue.put(sb)
            logger.info(f"[E2BPool] Sandbox {i} ready: {getattr(sb, 'id', None)}")
        logger.info("[E2BPool] All sandboxes ready.")

    async def acquire(self) -> Sandbox:
        """Acquire a sandbox from the pool."""
        sb = await self._queue.get()
        logger.debug(f"[E2BPool] Acquired sandbox {getattr(sb, 'id', None)}")
        return sb

    async def release(self, sb: Sandbox):
        """Return a sandbox to the pool."""
        await self._queue.put(sb)
        logger.debug(f"[E2BPool] Released sandbox {getattr(sb, 'id', None)}")

    async def shutdown(self):
        """Cleanly close all sandboxes."""
        logger.info("[E2BPool] Cleaning up sandboxes...")
        for sb in self._sandboxes:
            try:
                await asyncio.to_thread(sb.close)
            except Exception as e:
                logger.warning(f"[E2BPool] Failed to close sandbox: {e}")
        logger.info("[E2BPool] All sandboxes closed.")


@dataclasses.dataclass
class E2BClient(base_client.SandboxClient):
    backend_name: str = "e2b"

    def __init__(self, pool: E2BPool | None=None) -> None:
        self._template_id = os.getenv("E2B_TEMPLATE_ID")
        self._api_key = os.getenv("E2B_API_KEY")
        if not self._api_key:
            raise ValueError("Missing E2B_API_KEY environment variable.")
        if pool is None:
            self._pool = E2BPool(size=2)
            # start it asynchronously in the background if possible
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # running inside FastAPI/pytest async context
                loop.create_task(self._pool.start())
            else:
                # synchronous environment (rare)
                asyncio.run(self._pool.start())
        else:
            self._pool = pool

    async def run(self, code: str, tests: str) -> base_client.ExecutionResult:
        """Execute code and tests inside a pooled E2B sandbox."""
        payload = f"{code}\n\n{tests}"
        start = time.perf_counter()

        sb = await self._pool.acquire()
        try:
            result = await asyncio.to_thread(sb.run_code, payload)
            success = result["exit_code"] == 0
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
        except Exception as e:
            success = False
            stdout = ""
            stderr = str(e)
        finally:
            await self._pool.release(sb)

        runtime_ms = (time.perf_counter() - start) * 1000
        return base_client.ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            success=success,
            runtime_ms=runtime_ms,
            backend=self.backend_name,
            metadata={"sandbox_id": getattr(sb, "id", None)},
        )

    def close(self) -> None:
        """Gracefully stop the sandbox when done."""
        try:
            self._sandbox.close()
        except Exception:
            pass
