"""FastAPI thin layer for PolySandbox evaluation."""

from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional

from adapters.daytona_client import DaytonaClient, DaytonaPool, base_client
from evaluators import scorer, benchmark
from adapters.base_client import ExecutionResult
from agents import martian_agent

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
async def run_single(index: int = 0):
    """Run a single MBPP problem end-to-end and return detailed results."""
    global _pool
    if _pool is None:
        raise HTTPException(status_code=503, detail="Daytona pool not initialized")

    try:
        # Reuse existing Daytona pool and shared Martian agent
        client = DaytonaClient(_pool)
        agent = martian_agent.MartianAgent()

        # Run the single benchmark problem
        result = await benchmark.run_one(index, client, agent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    # If an error was returned by the benchmark
    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {result['error']}")

    # Construct structured API response
    response = RunResponse(
        backend="daytona",
        success=bool(result.get("success", False)),
        runtime_ms=float(result.get("runtime_ms", 0.0)),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        score=bool(result.get("score", False)),
        metadata={"idx": result.get("idx"), "note": "benchmark.run_one"},
    )

    print(
        f"[Result] idx={result['idx']} "
        f"success={result['success']} "
        f"score={'PASS' if result['score'] else 'FAIL'} "
        f"runtime={result['runtime_ms']:.2f}ms"
    )
    return response

    
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
