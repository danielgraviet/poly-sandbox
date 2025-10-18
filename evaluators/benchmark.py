"""Batch benchmark runner for MBPP using Martian + Daytona."""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from evaluators.executor import _MBPP_PROMPT_TEMPLATE
from adapters.daytona_client import DaytonaClient, base_client, DaytonaPool
from agents.martian_agent import MartianAgent
from utils import utils
from evaluators.scorer import binary_score
from hf_datasets.mbpp_loader import get_problem


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

    try:
        # Generate code
        print(f"[GEN] idx={idx} Calling MartianAgent.generate_code()")
        raw = await agent.generate_code(prompt)
        print(f"[GEN-DONE] idx={idx}")

        code = utils.extract_python_code(raw) or raw
        print(f"[EXTRACT] idx={idx} → {len(code.splitlines())} lines")

        # Execute code
        print(f"[RUN] idx={idx} Calling DaytonaClient.run()")
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


async def run_mbpp_batch(
    n: int = 120,
    concurrency: int = _CONCURRENCY_LIMIT,
    pool: Optional[DaytonaPool] = None,
):
    """Run all MBPP problems with parallel sandbox execution."""
    own_pool = False
    if pool is None:
        pool = DaytonaPool(size=5)
        await pool.start()
        own_pool = True

    client = DaytonaClient(pool)
    agent = MartianAgent()
    print("AGENT:", agent)
    print("client:", client)
    print(f"Starting MBPP batch run for {n} problems with concurrency={concurrency}\n")

    sem = asyncio.Semaphore(concurrency)

    async def sem_task(idx: int):
        async with sem:
            return await run_one(idx, client, agent)

    start = time.perf_counter()
    tasks = [asyncio.create_task(sem_task(i)) for i in range(n)]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    out_path = Path("outputs/mbpp_results.jsonl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nCompleted {n} MBPP problems in {total_time:.1f}s ({total_time/n:.2f}s avg)")
    print(f"Results saved to {out_path}")

    if own_pool:
        await pool.close()

    return results
