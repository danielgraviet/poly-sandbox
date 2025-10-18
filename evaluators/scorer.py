"""Simple scoring logic for PolySandbox."""

from __future__ import annotations
from adapters import base_client


def binary_score(result: base_client.ExecutionResult) -> bool:
    """Return True if success flag or passing text appears in stdout."""
    if result.success:
        return True
    if any(marker in (result.stdout or "").lower() for marker in ["pass", "success",]):
        return True
    return False
