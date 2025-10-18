"""Dataset loader for MBPP (Mostly Basic Python Problems)."""

from __future__ import annotations
from typing import Any, Dict, TypedDict, List
import datasets
from utils import utils
import pytest


class MBPPDataPoint(TypedDict):
    """Structured MBPP problem representation."""
    prompt: str
    canonical_solution: str
    function_name: str
    test_list: List[str]


def get_problem(index: int, split: str = "train") -> MBPPDataPoint:
    """Return one MBPP problem entry with validated structure."""
    ds = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    if index < 0 or index >= len(ds):
        raise IndexError(f"Index {index} out of range (dataset length = {len(ds)})")

    sample = ds[index]
    fn_name = utils.extract_fn_name(sample.get("code", ""))

    if fn_name is None:
        raise ValueError(f"[MBPP] No function name found in canonical_solution at index {index}")

    return {
        "prompt": sample.get("text") or sample.get("prompt", ""),
        "canonical_solution": sample.get("code", ""),
        "function_name": fn_name,
        "test_list": sample.get("test_list", []),
    }


def load_mbpp(split: str = "train", subset: str = "sanitized") -> List[MBPPDataPoint]:
    """Load full MBPP dataset and convert each row to MBPPDataPoint."""
    ds = datasets.load_dataset("google-research-datasets/mbpp", subset, split=split)

    all_problems: List[MBPPDataPoint] = []
    for idx, sample in enumerate(ds):
        fn_name = utils.extract_fn_name(sample.get("code", "")) or f"function_{idx}"
        problem: MBPPDataPoint = {
            "prompt": sample.get("text") or sample.get("prompt", ""),
            "canonical_solution": sample.get("code", ""),
            "function_name": fn_name,
            "test_list": sample.get("test_list", []),
        }
        all_problems.append(problem)

    return all_problems


def summarize_dataset(ds: datasets.Dataset) -> Dict[str, Any]:
    """Return simple metadata summary for debugging."""
    return {
        "num_examples": len(ds),
        "columns": ds.column_names,
        "first_keys": list(ds[0].keys()) if len(ds) > 0 else [],
    }