"""FastAPI thin layer for PolySandbox evaluation."""

from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional

from adapters.daytona_client import DaytonaClient, DaytonaPool
from adapters.e2b_client import E2BClient, E2BPool
from evaluators import benchmark
from adapters.base_client import ExecutionResult
from agents import martian_agent

app = FastAPI(title="PolySandbox API", version="0.1.0")

# -------------------------------------------------------------------------
# Backend Registry
# -------------------------------------------------------------------------
_BACKEND_REGISTRY = {
    "daytona": {"client": DaytonaClient, "pool_class": DaytonaPool},
    "e2b": {"client": E2BClient, "pool_class": E2BPool},
}

# -------------------------------------------------------------------------
# Global Pools
# -------------------------------------------------------------------------
_pools: dict[str, object] = {}

# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------
class RunResponse(BaseModel):
    backend: str
    success: bool
    runtime_ms: float
    stdout: str
    stderr: str
    score: bool
    metadata: Optional[dict] = None


class BatchRunResponse(BaseModel):
    total: int
    passed: int
    failed: int
    accuracy: float
    results_path: str


# -------------------------------------------------------------------------
# Startup & Shutdown
# -------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Spin up reusable sandbox pools at startup."""
    global _pools

    for backend, cfg in _BACKEND_REGISTRY.items():
        pool_class = cfg["pool_class"]
        print(f"[Startup] Initializing {backend} pool...")
        try:
            pool = pool_class(size=3)
            await pool.start()
            _pools[backend] = pool
            print(f"[Startup] {backend} pool ready.")
        except Exception as e:
            print(f"[Startup] Failed to init {backend} pool: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all sandbox pools."""
    global _pools

    for backend, pool in _pools.items():
        print(f"[Shutdown] Cleaning up {backend} sandboxes...")
        try:
            await pool.close()
        except AttributeError:
            # E2BPool might have .shutdown() instead of .close()
            await pool.shutdown()
        except Exception as e:
            print(f"[Shutdown] Error cleaning {backend}: {e}")
        print(f"[Shutdown] {backend} sandboxes deleted.")

    _pools.clear()


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------
@app.get("/backends", response_model=list[str])
def list_backends():
    """List available backends."""
    return list(_BACKEND_REGISTRY.keys())


@app.post("/run", response_model=RunResponse)
async def run_single(index: int = 0, backend: str = "daytona"):
    """Run a single MBPP problem end-to-end using selected backend."""
    global _pools

    if backend not in _BACKEND_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")

    pool = _pools.get(backend)
    if not pool:
        raise HTTPException(status_code=503, detail=f"{backend} pool not initialized")

    client_cls = _BACKEND_REGISTRY[backend]["client"]
    client = client_cls(pool)

    try:
        agent = martian_agent.MartianAgent()
        result = await benchmark.run_one(index, client, agent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {result['error']}")

    return RunResponse(
        backend=backend,
        success=bool(result.get("success", False)),
        runtime_ms=float(result.get("runtime_ms", 0.0)),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        score=bool(result.get("score", False)),
        metadata={"idx": result.get("idx"), "note": f"run_one via {backend}"},
    )


@app.post("/run_batch", response_model=BatchRunResponse)
async def run_batch(n: int = 10, concurrency: int = 5, backend: str = "daytona"):
    """Run MBPP batch directly from Hugging Face and return summary results."""
    global _pools

    # Validate backend
    if backend not in _BACKEND_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")

    pool = _pools.get(backend)
    if not pool:
        raise HTTPException(status_code=503, detail=f"{backend} pool not initialized")

    try:
        results = await benchmark.run_mbpp_batch(
            n=n,
            concurrency=concurrency,
            pool=pool,
            backend=backend,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {e}")

    # --- Compute simple metrics ---
    total = len(results)
    passed = sum(1 for r in results if r.get("score") is True)
    failed = total - passed
    accuracy = round(passed / total, 4) if total > 0 else 0.0

    # --- Log summary for visibility ---
    print(
        f"[Summary:{backend.upper()}] total={total} "
        f"passed={passed} failed={failed} accuracy={accuracy*100:.1f}%"
    )

    # --- Construct response ---
    return BatchRunResponse(
        total=total,
        passed=passed,
        failed=failed,
        accuracy=accuracy,
        results_path=f"outputs/{backend}_mbpp_results.jsonl",
    )



# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
