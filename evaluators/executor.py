"""Unified evaluator executor for PolySandbox backends."""

from __future__ import annotations
import time
from typing import Any
from adapters import daytona_client
from adapters import base_client
from agents import martian_agent
from hf_datasets import mbpp_loader
from utils import utils
from evaluators import scorer


_MBPP_PROMPT_TEMPLATE = """You are an expert Python programmer.

Write a correct, efficient, and readable solution for the following task.

Problem:
{problem_text}

You must define a function with the **exact name**:

```python
def {function_name}(...):
    # your implementation here
``` 
"""


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


async def run_single_mbpp(idx: int = 0, backend: str = "daytona") -> dict:
    """Run one MBPP problem end-to-end with a live model + sandbox backend.

    Steps:
      1. Load MBPP problem
      2. Generate code with MartianAgent
      3. Extract clean Python code
      4. Execute code + tests via Daytona
      5. Compute pass/fail score
    """

    problem = mbpp_loader.get_problem(idx)
    prompt = _MBPP_PROMPT_TEMPLATE.format(
        problem_text=problem["prompt"],
        function_name=problem["function_name"],
    )
    tests = "\n".join(problem["test_list"])

    agent = martian_agent.MartianAgent()
    raw_response = await agent.generate_code(prompt)
    extracted_code = utils.extract_python_code(raw_response) or raw_response

    client = daytona_client.DaytonaClient()
    result: base_client.ExecutionResult = await client.run(extracted_code, tests)

    score = scorer.binary_score(result)

    return {
        "problem_index": idx,
        "backend": backend,
        "score": score,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "runtime_ms": result.runtime_ms,
        "success": result.success,
        "generated_code": extracted_code,
        "raw_response": raw_response,
    }

