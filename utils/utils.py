import re
from typing import Optional

def extract_python_code(text: str) -> Optional[str]:
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
