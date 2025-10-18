from agents import martian_agent
import pytest

@pytest.mark.martian
@pytest.mark.asyncio
async def test_single_model():
    m_agent = martian_agent.MartianAgent()
    raw_response = await m_agent.generate_code(
        prompt="Create a simple two sum python function. Return your solution in ```python ``` code blocks.",
    )
    print("raw_response: ", raw_response)
    assert raw_response is not None