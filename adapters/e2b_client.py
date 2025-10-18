"""E2B sandbox adapter for PolySandbox."""

from __future__ import annotations

import os
import time
import dataclasses
import asyncio
import logging
from typing import Any, Optional, List

from e2b_code_interpreter import Sandbox
import adapters.base_client as base_client

_LOGGER = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# E2B Pool (parallel sandboxes)
# -------------------------------------------------------------------------
class E2BPool:
    """Manage a reusable pool of E2B sandboxes."""

    def __init__(self, size: int = 3, template_id: Optional[str] = None):
        self._size = size
        self._template_id = template_id or os.getenv("E2B_TEMPLATE_ID")
        if not self._template_id:
            raise ValueError("Missing E2B_TEMPLATE_ID environment variable.")
        self._sem = asyncio.Semaphore(size)
        self._sandboxes: List[Sandbox] = []
        self._lock = asyncio.Lock()

    async def start(self):
        """Provision all sandboxes synchronously (E2B SDK must run on main thread)."""
        print(f"[E2BPool] Spinning up {self._size} sandboxes...")

        # Run the blocking sandbox creation loop in a separate thread — one per pool.
        # Each Sandbox.create() *must* run synchronously, not inside asyncio.to_thread itself.
        def _create_sandboxes():
            sandboxes = []
            for i in range(self._size):
                try:
                    sb = Sandbox.create(timeout=60)
                    sandboxes.append(sb)
                    print(f"[E2BPool] Sandbox {i} ready: {sb.sandbox_id}")
                except Exception as e:
                    print(f"[E2BPool] Failed to create sandbox {i}: {e}")
            return sandboxes

        self._sandboxes = await asyncio.to_thread(_create_sandboxes)
        print(f"[E2BPool] Ready sandboxes: {len(self._sandboxes)}")


    async def acquire(self) -> Sandbox:
        """Get an available sandbox."""
        await self._sem.acquire()
        async with self._lock:
            if not self._sandboxes:
                raise RuntimeError("No available E2B sandboxes in pool")
            return self._sandboxes.pop()

    async def release(self, sb: Sandbox):
        """Return sandbox to pool."""
        async with self._lock:
            self._sandboxes.append(sb)
        self._sem.release()

    async def close(self):
        """Kill all active sandboxes."""
        print("[E2BPool] Cleaning up sandboxes...")
        for sb in self._sandboxes:
            try:
                await asyncio.to_thread(sb.kill)
            except Exception as e:
                print("[E2BPool] Cleanup error:", e)
        self._sandboxes.clear()


# -------------------------------------------------------------------------
# E2B Client Adapter
# -------------------------------------------------------------------------
@dataclasses.dataclass
class E2BClient(base_client.SandboxClient):
    """Adapter for executing code securely via the E2B Sandbox SDK."""

    backend_name: str = "e2b"

    def __init__(self, pool: Optional[E2BPool] = None) -> None:
        # Auto-initialize pool if not provided
        self._pool = pool or E2BPool(size=1)
        if not pool:
            # If user didn't provide a pool, start it automatically
            asyncio.run(self._pool.start())

    async def run(self, code: str, tests: str) -> base_client.ExecutionResult:
        """Execute code and tests in E2B sandbox."""
        payload = f"{code}\n\n{tests}"
        sb = await self._pool.acquire()

        try:
            start = time.perf_counter()
            result = await asyncio.to_thread(sb.run_code, payload)
            runtime_ms = (time.perf_counter() - start) * 1000

            stdout = "".join(result.logs.stdout) if result.logs and result.logs.stdout else ""
            stderr = "".join(result.logs.stderr) if result.logs and result.logs.stderr else ""
            success = result.error is None

            return base_client.ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                success=success,
                runtime_ms=runtime_ms,
                backend=self.backend_name,
                metadata={
                    "sandbox_id": getattr(sb, "sandbox_id", None),
                    "template_id": getattr(sb, "template_id", None),
                },
            )

        finally:
            await self._pool.release(sb)
