"""Batch benchmark runner for MBPP using Martian + Daytona."""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

from evaluators.executor import _MBPP_PROMPT_TEMPLATE
from adapters.daytona_client import DaytonaClient, base_client
from agents.martian_agent import MartianAgent
from utils.utils import extract_python_code
from evaluators.scorer import binary_score
from hf_datasets.mbpp_loader import get_problem


_CONCURRENCY_LIMIT = 20  # number of sandboxes / parallel tasks


async def run_one(idx: int, client: base_client.SandboxClient, agent: MartianAgent) -> Dict[str, Any]:
    """Run one MBPP problem end-to-end asynchronously."""
    print(f"\n[START] Problem {idx}")
    problem = get_problem(idx)
    prompt = _MBPP_PROMPT_TEMPLATE.format(problem_text=problem["prompt"])
    tests = "\n".join(problem["test_list"])

    try:
        # Generate code
        print(f"[GEN] idx={idx} Calling MartianAgent.generate_code()")
        raw = await agent.generate_code(prompt)
        print(f"[GEN-DONE] idx={idx}")

        code = extract_python_code(raw) or raw
        print(f"[EXTRACT] idx={idx} → {len(code.splitlines())} lines")

        # Execute code
        print(f"[RUN] idx={idx} Calling DaytonaClient.run()")
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


async def run_mbpp_batch(n: int = 120, concurrency: int = _CONCURRENCY_LIMIT):
    """Run all MBPP problems with parallel sandbox execution."""
    agent = MartianAgent()
    client = DaytonaClient()
    print("AGENT: ", agent) # showing 
    print("client: ", client) # showing
    print(f"Starting MBPP batch run for {n} problems with concurrency={concurrency}\n")

    sem = asyncio.Semaphore(concurrency) 

    # could be doing the full evalution, but i want you to add logs so I can see each problem 
    async def sem_task(idx: int):
        async with sem:
            return await run_one(idx, client, agent)

    start = time.perf_counter()
    tasks = [asyncio.create_task(sem_task(i)) for i in range(n)]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    # Save results
    out_path = Path("outputs/mbpp_results.jsonl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nCompleted {n} MBPP problems in {total_time:.1f}s ({total_time/n:.2f}s per problem avg)")
    print(f"Results saved to {out_path}")

    return results
