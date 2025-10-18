import re
from typing import Optional, Dict, List
from pathlib import Path
import json

def extract_python_code(text: str) -> Optional[str]:
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_fn_name(code: str) -> str | None:
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    return match.group(1) if match else None


def summarize_mbpp_results(path: str | Path) -> None:
    """Print overall MBPP benchmark accuracy and simple stats."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r") as f:
        results: List[Dict] = [json.loads(line) for line in f if line.strip()]

    total = len(results)
    passed = sum(1 for r in results if r.get("score") is True)
    failed = total - passed
    pct = (passed / total * 100) if total > 0 else 0.0

    print("\nMBPP Benchmark Summary")
    print("──────────────────────────────")
    print(f"Total problems: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Accuracy: {pct:.1f}%\n")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": pct,
    }
