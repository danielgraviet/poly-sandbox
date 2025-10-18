
"""Daytona sandbox adapter for PolySandbox."""

from __future__ import annotations

import os
import time
import dataclasses
import asyncio
from typing import Any, Optional, List

from daytona import Daytona, DaytonaConfig
import adapters.base_client as base_client
import logging

_LOGGER = logging.getLogger(__name__)

class DaytonaPool:
    """Manage a reusable pool of Daytona sandboxes."""

    def __init__(self, size: int = 5, api_key: Optional[str] = None):
        self._size = size
        self._api_key = api_key or os.getenv("DAYTONA_API_KEY")
        self._daytona = Daytona(DaytonaConfig(api_key=self._api_key))
        self._sem = asyncio.Semaphore(size)
        self._sandboxes: List = []
        self._lock = asyncio.Lock()

    async def start(self):
        """Provision all sandboxes and wait until ready."""
        print(f"[Pool] Spinning up {self._size} sandboxes...")
        for i in range(self._size):
            try:
                sb = self._daytona.create()
                sb_id = getattr(sb, "id", None)

                # Give Daytona a brief moment to finish booting
                await asyncio.sleep(2)

                self._sandboxes.append(sb)
                print(f"[Pool] Sandbox {i} ready: {sb_id}")

            except Exception as e:
                print(f"[Pool] Failed to create sandbox {i}: {e}")            
        print("[Pool] All sandboxes ready.")

    async def acquire(self):
        """Get a sandbox for use."""
        await self._sem.acquire()
        async with self._lock:
            return self._sandboxes.pop()

    async def release(self, sb):
        """Return a sandbox back to pool."""
        async with self._lock:
            self._sandboxes.append(sb)
        self._sem.release()

    async def close(self):
        """Clean up all sandboxes."""
        print("[Pool] Cleaning up sandboxes...")
        for sb in self._sandboxes:
            try:
                sb.delete()
            except Exception as e:
                print("[Pool] Cleanup error:", e)
        self._sandboxes.clear()



@dataclasses.dataclass
class DaytonaClient(base_client.SandboxClient):
    """Adapter for executing code securely via the Daytona Sandbox API."""

    backend_name: str = "daytona"

    def __init__(self, pool: DaytonaPool) -> None:
        self._pool = pool

    async def run(self, code: str, tests: str) -> base_client.ExecutionResult:
        """Executes code and test suite inside a Daytona sandbox."""

        # Combine code + tests into one execution payload
        payload = f"{code}\n\n{tests}"
        sb = await self._pool.acquire()

        try:
            start = time.perf_counter()
            resp = sb.process.code_run(payload)
            success = resp.exit_code == 0
            runtime_ms = (time.perf_counter() - start) * 1000
            return base_client.ExecutionResult(
                stdout=resp.result if success else "",
                stderr="" if success else resp.result,
                success=success,
                runtime_ms=runtime_ms,
                backend=self.backend_name,
                metadata={"sandbox_id": getattr(sb, "id", None)},
            )
        finally:
            await self._pool.release(sb)
