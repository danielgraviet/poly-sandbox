"""Simple MVP RunnerAgent for PolySandbox — auto-selects backend and runs MBPP problem."""

from dataclasses import dataclass
from evaluators import benchmark
from hf_datasets.mbpp_loader import get_problem
from agents.martian_agent import MartianAgent
from adapters.daytona_client import DaytonaClient, DaytonaPool
from adapters.e2b_client import E2BClient, E2BPool


@dataclass
class RunnerAgent:
    """Chooses backend (based on code length) and runs a single MBPP problem."""

    daytona_pool: DaytonaPool
    e2b_pool: E2BPool

    async def run_problem(self, index: int = 0):
        """Load MBPP problem, decide backend, and run it end-to-end."""
        # --- 1. Load problem from MBPP dataset ---
        problem = get_problem(index)
        prompt = benchmark._MBPP_PROMPT_TEMPLATE.format(
            problem_text=problem["prompt"],
            function_name=problem["function_name"],
        )
        tests = "\n".join(problem["test_list"])

        # --- 2. Decide backend based on code length ---
        backend = "daytona" if len(prompt) < 500 else "e2b"
        print(f"[RunnerAgent] Selected backend: {backend}")

        # --- 3. Initialize backend client + agent ---
        client = DaytonaClient(self.daytona_pool) if backend == "daytona" else E2BClient(self.e2b_pool)
        agent = MartianAgent()

        # --- 4. Run the problem through your evaluation harness ---
        result = await benchmark.run_one(index, client, agent)

        # --- 5. Return structured result for UI ---
        return {
            "backend": backend,
            "idx": index,
            "success": result.get("success", False),
            "score": result.get("score", False),
            "runtime_ms": result.get("runtime_ms", 0.0),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
