import pytest
import asyncio
from adapters import base_client

class MockClient(base_client.SandboxClient):
    backend_name = "mock"

    async def run(self, code: str, tests: str) -> base_client.ExecutionResult:
        return base_client.ExecutionResult(
            stdout="ok",
            stderr="",
            success=True,
            runtime_ms=0.0,
            backend=self.backend_name,
        )


@pytest.mark.asyncio
async def test_execution_result_fields():
    res = base_client.ExecutionResult(
        stdout="out",
        stderr="",
        success=True,
        runtime_ms=12.3,
        backend="mock",
    )
    assert res.stdout == "out"
    assert res.backend == "mock"
    assert isinstance(res.metadata, dict)


@pytest.mark.asyncio
async def test_mock_client_run():
    client = MockClient()
    result = await client.timed_run("print('hi')", "assert 1==1")
    assert result.success
    assert "mock" in result.backend
    assert result.runtime_ms >= 0
