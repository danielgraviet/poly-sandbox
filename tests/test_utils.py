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


_EXAMPLE_CODE = """def noprofit_noloss(actual_cost, sale_amount):
    if sale_amount == actual_cost:
        return True
    else:
        return False
"""

@pytest.mark.utils
def test_extract_function_def():
    fn = utils.extract_fn_name(_EXAMPLE_CODE)
    print("Extracted fn name:", fn)
    assert fn == "noprofit_noloss"