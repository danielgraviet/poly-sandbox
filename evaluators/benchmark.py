"""Batch benchmark runner for MBPP using Martian + Daytona."""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import random

from evaluators.executor import _MBPP_PROMPT_TEMPLATE
from adapters.daytona_client import DaytonaClient, base_client, DaytonaPool
from adapters.e2b_client import E2BClient, E2BPool
from agents.martian_agent import MartianAgent
from utils import utils
from evaluators.scorer import binary_score
from hf_datasets.mbpp_loader import get_problem
_last_backend = None  # global tracker



_CONCURRENCY_LIMIT = 5  # number of sandboxes / parallel tasks


async def run_one(idx: int, client: base_client.SandboxClient, agent: MartianAgent) -> Dict[str, Any]:
    """Run one MBPP problem end-to-end asynchronously."""
    print(f"\n[START] Problem {idx}")
    problem = get_problem(idx)
    prompt = _MBPP_PROMPT_TEMPLATE.format(
        problem_text=problem["prompt"],
        function_name=problem["function_name"],
    )
    print("PROMPT: ", prompt)
    tests = "\n".join(problem["test_list"])
    if "E2BClient" in str(client):
        print("e2b sleep!")
        await asyncio.sleep(1.0)

    try:
        # Generate code
        print(f"[GEN] idx={idx} Calling MartianAgent.generate_code()")
        raw = await agent.generate_code(prompt)
        print(f"[GEN-DONE] idx={idx}")

        code = utils.extract_python_code(raw) or raw
        print(f"[EXTRACT] idx={idx} → {len(code.splitlines())} lines")

        # Execute code
        print(f"[RUN] idx={idx} Calling client.run()")
        print(f"[DEBUG] idx={idx} --- EXECUTING CODE ---\n{code}\n")
        print(f"[DEBUG] idx={idx} --- EXECUTING TESTS ---\n{tests}\n")
        result = await client.run(code, tests)
        print(f"[RUN-DONE] idx={idx} success={result.success}")
        
        score = binary_score(result)
        print(f"[SCORE] idx={idx} → {'PASS' if score else 'FAIL'}")

        return {
            "idx": idx,
            "success": result.success,
            "score": score,
            "runtime_ms": result.runtime_ms,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "code": code,
        }

    except Exception as e:
        return {"idx": idx, "error": str(e)}


async def run_agent(index: int = 0, pools: Optional[dict] = None):
    """Autonomous agent that decides backend (E2B vs Daytona) based on prompt length."""
    print(f"\n[AGENT] Starting autonomous run for problem #{index}")
    problem = get_problem(index)
    prompt = _MBPP_PROMPT_TEMPLATE.format(
        problem_text=problem["prompt"],
        function_name=problem["function_name"],
    )

    # --- Decision logic (MVP heuristic) ---
    global _last_backend

    # Toggle between Daytona and E2B each call
    if _last_backend == "daytona":
        backend = "e2b"
    else:
        backend = "daytona"

    _last_backend = backend
    print(f"[AGENT] Toggled backend → {backend}")

    if backend == "e2b":
        await asyncio.sleep(1.0)
    print(f"[AGENT] Selected backend: {backend} (prompt length={len(prompt)})")

    if not pools or backend not in pools:
        raise RuntimeError(f"{backend} pool not initialized in server")

    pool = pools[backend]
    client_cls = DaytonaClient if backend == "daytona" else E2BClient
    client = client_cls(pool)
    agent = MartianAgent()

    # --- Run evaluation ---
    result = await run_one(index, client, agent)
    result["backend"] = backend
    print(
        f"[AGENT] Completed problem #{index} "
        f"→ backend={backend}, success={result.get('success')}, score={result.get('score')}"
    )

    return result


async def run_mbpp_batch(
    n: int = 120,
    concurrency: int = _CONCURRENCY_LIMIT,
    pool: Optional[Any] = None,
    backend: str = "daytona",
):
    """Run all MBPP problems with parallel sandbox execution (backend-agnostic)."""

    # Select appropriate client/pool based on backend
    if backend == "daytona":
        client_cls, pool_cls = DaytonaClient, DaytonaPool
    elif backend == "e2b":
        client_cls, pool_cls = E2BClient, E2BPool
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Spin up our own pool if none provided
    own_pool = False
    if pool is None:
        pool = pool_cls(size=min(concurrency, 5))
        await pool.start()
        own_pool = True

    # Initialize client + agent
    client = client_cls(pool)
    if "E2BClient" in str(client):
        print("e2b sleep!")
        await asyncio.sleep(1.0)
    agent = MartianAgent()
    print(f"[INIT] Backend={backend}")
    print(f"[INIT] Agent={agent}")
    print(f"[INIT] Client={client}")
    print(f"[INIT] Running {n} MBPP problems with concurrency={concurrency}\n")

    # Semaphore for concurrency limiting
    sem = asyncio.Semaphore(concurrency)

    async def sem_task(idx: int):
        async with sem:
            return await run_one(idx, client, agent)

    # Launch tasks
    start = time.perf_counter()
    tasks = [asyncio.create_task(sem_task(i)) for i in range(n)]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    # Save results per backend
    out_path = Path(f"outputs/{backend}_mbpp_results.jsonl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(
        f"\n[{backend.upper()}] Completed {n} MBPP problems "
        f"in {total_time:.1f}s ({total_time/n:.2f}s avg)"
    )
    print(f"[{backend.upper()}] Results saved to {out_path}")

    # Cleanup if pool was owned here
    if own_pool:
        await pool.close() if hasattr(pool, "close") else await pool.shutdown()

    return results