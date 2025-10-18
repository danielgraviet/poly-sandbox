"""Dataset loader for MBPP (Mostly Basic Python Problems)."""

from __future__ import annotations
from typing import Any, Dict
import datasets


def load_mbpp(split: str = "train", subset: str = "sanitized") -> datasets.Dataset:
    return datasets.load_dataset("google-research-datasets/mbpp", subset, split=split)


def get_problem(index: int, split: str = "train") -> Dict[str, Any]:
    """Return one MBPP problem entry.

    Args:
        index: The integer index of the problem (0–119 for sanitized train).
        split: Which split to pull from ("train" by default).

    Returns:
        dict with keys: prompt, canonical_solution, test_list, and optionally starter_code.
    """
    ds = load_mbpp(split=split)
    if index < 0 or index >= len(ds):
        raise IndexError(f"Index {index} out of range (dataset length = {len(ds)})")

    sample = ds[index]
    return {
        "prompt": sample.get("text") or sample.get("prompt"),
        "canonical_solution": sample.get("code") or sample.get("canonical_solution"),
        "test_list": sample.get("test_list", []),
        "starter_code": sample.get("starter_code", ""),
    }


def summarize_dataset(ds: datasets.Dataset) -> Dict[str, Any]:
    """Return simple metadata summary for debugging."""
    return {
        "num_examples": len(ds),
        "columns": ds.column_names,
        "first_keys": list(ds[0].keys()) if len(ds) > 0 else [],
    }
