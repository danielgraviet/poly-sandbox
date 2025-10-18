from __future__ import annotations
import abc
import time
import dataclasses 
from typing import Any, Optional


@dataclasses.dataclass
class ExecutionResult:
    """Normalized result returned from any sandbox backend."""
    stdout: str
    stderr: str
    success: bool
    runtime_ms: float
    backend: str
    metadata: Optional[dict[str, Any]]


class SandboxClient(abc.ABC):
    """Abstract interface all sandbox adapters (Daytona, E2B, Docker) must follow."""

    backend_name: str

    @abc.abstractmethod
    async def run(self, code: str, tests: str) -> ExecutionResult:
        """Run given code and tests in the sandbox, return structured result."""
        raise NotImplementedError

    async def timed_run(self, code: str, tests: str) -> ExecutionResult:
        """Utility wrapper to measure runtime_ms automatically."""
        start = time.perf_counter()
        result = await self.run(code, tests)
        end = time.perf_counter()
        result.runtime_ms = (end - start) * 1000
        return result
