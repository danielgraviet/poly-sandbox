"""Unified evaluator executor for PolySandbox backends."""

from __future__ import annotations
import time
from typing import Any
from adapters import daytona_client
from adapters import base_client

# later you can import E2BClient, DockerClient here

_BACKEND_REGISTRY = {
    "daytona": daytona_client.DaytonaClient,
    # "e2b": E2BClient,
    # "docker": DockerClient,
}


async def run_pipeline(backend: str, code: str, tests: str) -> base_client.ExecutionResult:
    """Run code/tests pair on selected backend and return normalized ExecutionResult."""

    if backend not in _BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend '{backend}'. Available: {list(_BACKEND_REGISTRY)}")

    client_cls = _BACKEND_REGISTRY[backend]
    client = client_cls()

    start = time.perf_counter()
    result = await client.run(code=code, tests=tests)
    end = time.perf_counter()

    # If backend didn’t populate runtime_ms, compute fallback
    if not getattr(result, "runtime_ms", None):
        result.runtime_ms = (end - start) * 1000

    return result
