import pytest
from utils import utils

@pytest.mark.utils
def test_extract_python_code():
    text = "```python\nprint('Hello World')\n```"
    assert utils.extract_python_code(text) == "print('Hello World')"

@pytest.mark.utils
def test_extract_missing_code():
    text = "No code here."
    assert utils.extract_python_code(text) is None
