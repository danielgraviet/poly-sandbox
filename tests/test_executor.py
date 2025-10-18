import os
import pytest
import datasets

from evaluators.executor import run_pipeline
from evaluators.scorer import binary_score
from adapters.base_client import ExecutionResult
from hf_datasets import mbpp_loader


@pytest.mark.asyncio
async def test_run_pipeline_mock(monkeypatch):
    """Mock the backend to ensure glue layer logic works."""

    class MockClient:
        async def run(self, code: str, tests: str) -> ExecutionResult:
            return ExecutionResult(
                stdout="All tests passed",
                stderr="",
                success=True,
                runtime_ms=42.0,
                backend="mock",
            )

    monkeypatch.setattr("evaluators.executor._BACKEND_REGISTRY", {"mock": MockClient})

    result = await run_pipeline("mock", "print('hi')", "assert 1 == 1")
    assert isinstance(result, ExecutionResult)
    assert binary_score(result) is True

@pytest.mark.daytona
@pytest.mark.asyncio
async def test_run_pipeline_with_daytona_mbpp():
    """Live test on MBPP dataset (skips if no Daytona credentials)."""

    if not os.getenv("DAYTONA_API_KEY"):
        pytest.skip("Missing DAYTONA_API_KEY — skipping live Daytona test")

    # Load one MBPP sample
    ds = mbpp_loader.load_mbpp()
    sample = ds[0]

    code = sample["code"] if "code" in sample else sample["canonical_solution"]
    tests = "\n".join(sample["test_list"])
    print("CODE: ", code)
    
    single_problem = mbpp_loader.get_problem(
        index = 1,
        split = "train"
    )
    print("\nsingle_problem: ", single_problem)
    
    result = await run_pipeline("daytona", code, tests)

    print("\n--- Daytona MBPP Result ---")
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    print("runtime:", result.runtime_ms, "ms")
    print("result obj: ", result)

    assert isinstance(result, ExecutionResult)
    assert "daytona" in result.backend
