"""FastAPI thin layer for PolySandbox evaluation."""

from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional

from adapters.daytona_client import DaytonaClient, DaytonaPool, base_client
from evaluators import scorer, benchmark
from adapters.base_client import ExecutionResult

app = FastAPI(title="PolySandbox API", version="0.1.0")

# -------------------------------------------------------------------------
# Backend Registry
# -------------------------------------------------------------------------
_BACKEND_REGISTRY = {
    "daytona": DaytonaClient,
    # "e2b": E2BClient,
    # "docker": DockerClient,
}

# -------------------------------------------------------------------------
# Global Pool Initialization
# -------------------------------------------------------------------------
_pool: Optional[DaytonaPool] = None

class BatchRunResponse(BaseModel):
    total: int
    passed: int
    failed: int
    accuracy: float
    results_path: str


@app.on_event("startup")
async def startup_event():
    """Spin up reusable Daytona pool at startup."""
    global _pool
    print("[Startup] Initializing Daytona pool...")
    _pool = DaytonaPool(size=5)
    await _pool.start()
    print("[Startup] Daytona pool ready.")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up sandboxes on shutdown."""
    global _pool
    if _pool:
        print("[Shutdown] Cleaning up sandboxes...")
        await _pool.close()
        print("[Shutdown] All sandboxes deleted.")


# -------------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------------

class RunRequest(BaseModel):
    backend: Literal["daytona", "e2b", "docker"] = Field(..., description="Execution backend")
    code: str = Field(..., description="Python code to execute")
    tests: str = Field(..., description="Test code to validate correctness")


class RunResponse(BaseModel):
    backend: str
    success: bool
    runtime_ms: float
    stdout: str
    stderr: str
    score: bool
    metadata: Optional[dict] = None


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.get("/backends", response_model=list[str])
def list_backends():
    """List available execution backends."""
    return list(_BACKEND_REGISTRY.keys())


@app.post("/run", response_model=RunResponse)
async def run_code(req: RunRequest):
    """Run code + tests inside a sandbox backend."""
    global _pool

    if req.backend not in _BACKEND_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {req.backend}")

    if _pool is None:
        raise HTTPException(status_code=503, detail="Daytona pool not initialized")

    backend_cls = _BACKEND_REGISTRY[req.backend]
    try:
        client = backend_cls(_pool)  # ✅ pass in pool
        result: ExecutionResult = await client.run(req.code, req.tests)
        score = scorer.binary_score(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    return RunResponse(
        backend=req.backend,
        success=result.success,
        runtime_ms=result.runtime_ms,
        stdout=result.stdout,
        stderr=result.stderr,
        score=score,
        metadata=result.metadata,
    )
    
@app.post("/run_batch", response_model=BatchRunResponse)
async def run_batch(n: int = 10, concurrency: int = 5):
    """Run MBPP batch directly from Hugging Face and return summary results."""
    global _pool
    if _pool is None:
        raise HTTPException(status_code=503, detail="Daytona pool not initialized")

    try:
        # Run full evaluation but reuse the existing Daytona pool
        results = await benchmark.run_mbpp_batch(n=n, concurrency=concurrency, pool=_pool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {e}")

    # Compute summary stats
    total = len(results)
    passed = sum(1 for r in results if r.get("score") is True)
    failed = total - passed
    accuracy = round(passed / total, 4) if total > 0 else 0.0

    # Construct API response for frontend consumption
    summary = BatchRunResponse(
        total=total,
        passed=passed,
        failed=failed,
        accuracy=accuracy,
        results_path="outputs/mbpp_results.jsonl",
    )

    print(f"[Summary] total={total} passed={passed} failed={failed} accuracy={accuracy*100:.1f}%")
    return summary




# -------------------------------------------------------------------------
# Entry point for local dev
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
