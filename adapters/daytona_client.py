
"""Daytona sandbox adapter for PolySandbox."""

from __future__ import annotations

import os
import time
import dataclasses
from typing import Any

from daytona import Daytona, DaytonaConfig
import adapters.base_client as base_client


@dataclasses.dataclass
class DaytonaClient(base_client.SandboxClient):
    """Adapter for executing code securely via the Daytona Sandbox API."""

    backend_name: str = "daytona"

    def __init__(self) -> None:
        api_key = os.getenv("DAYTONA_API_KEY")
        if not api_key:
            raise ValueError("Missing DAYTONA_API_KEY environment variable.")

        config = DaytonaConfig(api_key=api_key)
        self._daytona = Daytona(config=config)
        self._sandbox = self._daytona.create()

    async def run(self, code: str, tests: str) -> base_client.ExecutionResult:
        """Executes code and test suite inside a Daytona sandbox."""

        # Combine code + tests into one execution payload
        payload = f"{code}\n\n{tests}"

        start = time.perf_counter()
        response = self._sandbox.process.code_run(payload)
        runtime_ms = (time.perf_counter() - start) * 1000

        success = response.exit_code == 0
        stdout = response.result if success else ""
        stderr = "" if success else response.result

        # Clean up the sandbox session after execution
        self._sandbox.delete()

        return base_client.ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            success=success,
            runtime_ms=runtime_ms,
            backend=self.backend_name,
            metadata={
                "exit_code": response.exit_code,
                "sandbox_id": getattr(self._sandbox, "id", None),
            },
        )
