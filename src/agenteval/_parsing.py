"""Helpers for turning loosely-formatted model output into Python values."""

from __future__ import annotations

import json
from typing import Any


def as_text(value: Any, default: str = "") -> str:
    """Models sometimes return a list or null where a string is expected. Normalize it."""
    if value is None:
        return default
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def _strip_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def extract_json(text: str, expect: type) -> Any:
    """Parse the first JSON array/object in ``text``.

    Tolerates code fences and prose before/after the JSON. Raises ``ValueError`` if none is found.
    """
    text = _strip_fences(text)
    try:
        value = json.loads(text)
        if isinstance(value, expect):
            return value
    except json.JSONDecodeError:
        pass

    open_ch = "[" if expect is list else "{"
    decoder = json.JSONDecoder()
    idx = text.find(open_ch)
    while idx != -1:
        try:
            value, _ = decoder.raw_decode(text[idx:])
            if isinstance(value, expect):
                return value
        except json.JSONDecodeError:
            pass
        idx = text.find(open_ch, idx + 1)
    raise ValueError(f"Could not find a JSON {expect.__name__} in model output: {text[:200]!r}")
