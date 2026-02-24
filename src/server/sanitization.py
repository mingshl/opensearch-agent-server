"""Input sanitization for user-provided data.

This module provides security-oriented sanitization for user-controlled input
that flows into tool results, state snapshots, and related structures. It
complements Pydantic validation by:

- Stripping control characters and dangerous Unicode (e.g. bidirectional overrides)
- Enforcing depth and size limits to prevent DoS (stack overflow, memory exhaustion)
- Validating dict keys to reduce injection and parsing surprises

**When to use:**
- State snapshots: `sanitize_for_state_snapshot(value)` before validation
- Tool result data: `sanitize_for_tool_result(value)` after parsing
- Message content (optional): `strip_control_chars(s)` for display/log safety

**Design:**
- Preserves JSON-serializable primitives (int, float, bool, None) as-is
- Recursively sanitizes dict/list/str with configurable limits
- Drops or replaces non-serializable and over-limit values; callers can log
"""

from __future__ import annotations

import re
from typing import Any

# --- Constants ---------------------------------------------------------------

# ASCII control chars except tab, newline, carriage return
# \x00-\x08: NUL and controls; \x0b\x0c: VT, FF; \x0e-\x1f: Shift Out to US; \x7f: DEL
_CONTROL_ASCII = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Unicode bidirectional override chars (can be used for visual spoofing)
_BIDI_OVERRIDES = re.compile(r"[\u202a-\u202e\u2066-\u2069]")

# Safe key: alphanumeric, underscore, hyphen, period. Max length.
_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,256}$")

# Sentinel for values to drop (non-JSON-serializable types)
_DROP = object()

# Limits for state snapshots (stricter)
MAX_STRING_LENGTH_STATE = 100_000  # 100 KB
MAX_DEPTH_STATE = 20
MAX_DICT_KEYS_STATE = 1000
MAX_LIST_LEN_STATE = 10_000

# Limits for tool results (tool output can be larger, e.g. logs)
MAX_STRING_LENGTH_TOOL_RESULT = 500_000  # 500 KB
MAX_DEPTH_TOOL_RESULT = 25
MAX_DICT_KEYS_TOOL_RESULT = 2000
MAX_LIST_LEN_TOOL_RESULT = 20_000


# --- String helpers ---------------------------------------------------------


def strip_control_chars(s: str) -> str:
    """Remove ASCII control chars and Unicode bidi overrides from a string.

    Preserves tab, newline, and carriage return. Removes:
    - ASCII 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F
    - Unicode U+202A–U+202E, U+2066–U+2069 (bidirectional overrides)

    Args:
        s: Input (str, or None/other types: None -> "", non-str -> str(s))

    Returns:
        String with control and bidi override chars removed

    """
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    t = _CONTROL_ASCII.sub("", s)
    t = _BIDI_OVERRIDES.sub("", t)
    return t


def truncate_string(s: str, max_length: int) -> str:
    """Truncate string to max_length. No-op if within limit.

    Args:
        s: Input string
        max_length: Maximum length (>= 0)

    Returns:
        Truncated string (with no ellipsis; caller may add if needed)

    """
    if not isinstance(s, str) or max_length < 0:
        return s if isinstance(s, str) else ""
    return s[:max_length] if len(s) > max_length else s


def is_safe_key(key: Any) -> bool:
    """Return True if key is acceptable for state/tool-result dicts.

    Allows alphanumeric, underscore, hyphen, period; length 1–256.

    Args:
        key: Dict key (must be str)

    Returns:
        True if key matches allowed pattern

    """
    if not isinstance(key, str) or len(key) == 0:
        return False
    return bool(_KEY_PATTERN.match(key))


def sanitize_key(key: Any) -> str | None:
    """Return sanitized key if valid; None if invalid (caller should drop).

    Args:
        key: Dict key

    Returns:
        key as str if is_safe_key, else None

    """
    if not isinstance(key, str) or len(key) == 0:
        return None
    k = key[:256] if len(key) > 256 else key
    if _KEY_PATTERN.match(k):
        return k
    return None


# --- Recursive sanitization -------------------------------------------------


def _sanitize_value(
    value: Any,
    depth: int,
    *,
    max_depth: int,
    max_string_length: int,
    max_dict_keys: int,
    max_list_len: int,
) -> Any:
    """Recursively sanitize a value. Internal; use preset helpers below."""
    if depth > max_depth:
        return None

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        cleaned = strip_control_chars(value)
        return truncate_string(cleaned, max_string_length)

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= max_dict_keys:
                break
            sk = sanitize_key(k)
            if sk is None:
                continue
            sv = _sanitize_value(
                v,
                depth + 1,
                max_depth=max_depth,
                max_string_length=max_string_length,
                max_dict_keys=max_dict_keys,
                max_list_len=max_list_len,
            )
            if sv is _DROP:
                continue
            out[sk] = sv
        return out

    if isinstance(value, (list, tuple)):
        lst = list(value)[:max_list_len]
        out_list: list[Any] = []
        for item in lst:
            v = _sanitize_value(
                item,
                depth + 1,
                max_depth=max_depth,
                max_string_length=max_string_length,
                max_dict_keys=max_dict_keys,
                max_list_len=max_list_len,
            )
            if v is not _DROP:
                out_list.append(v)
        return out_list

    # Non-JSON-serializable types: drop (caller may warn)
    return _DROP


def sanitize_for_state_snapshot(value: Any) -> Any:
    """Sanitize a value for use in state snapshots.

    Applies:
    - strip_control_chars and truncation for strings (max 100 KB)
    - depth limit 20, max 1000 dict keys, max 10k list items
    - safe-key check for dict keys; drops invalid keys
    - preserves int, float, bool, None

    Use this before or as part of validate_state_snapshot.

    Args:
        value: State value (typically a dict); can be any JSON-like structure

    Returns:
        Sanitized, JSON-serializable structure

    """
    return _sanitize_value(
        value,
        0,
        max_depth=MAX_DEPTH_STATE,
        max_string_length=MAX_STRING_LENGTH_STATE,
        max_dict_keys=MAX_DICT_KEYS_STATE,
        max_list_len=MAX_LIST_LEN_STATE,
    )


def sanitize_for_tool_result(value: Any) -> Any:
    """Sanitize parsed tool result data before emitting to events/hooks.

    Same rules as state but with looser limits (500 KB strings, depth 25,
    2000 dict keys, 20k list items) to support larger tool output (e.g. logs).

    Use after parse_tool_result_data / parse_json_with_fallback and before
    putting result_data into ToolMessage, state_from_result, etc.

    Args:
        value: Parsed tool result (dict, list, str, or primitive)

    Returns:
        Sanitized, JSON-serializable structure

    """
    return _sanitize_value(
        value,
        0,
        max_depth=MAX_DEPTH_TOOL_RESULT,
        max_string_length=MAX_STRING_LENGTH_TOOL_RESULT,
        max_dict_keys=MAX_DICT_KEYS_TOOL_RESULT,
        max_list_len=MAX_LIST_LEN_TOOL_RESULT,
    )


def sanitize_value(
    value: Any,
    *,
    max_depth: int = MAX_DEPTH_STATE,
    max_string_length: int = MAX_STRING_LENGTH_STATE,
    max_dict_keys: int = MAX_DICT_KEYS_STATE,
    max_list_len: int = MAX_LIST_LEN_STATE,
) -> Any:
    """Sanitize with explicit limits. Prefer sanitize_for_state_snapshot or sanitize_for_tool_result.

    Args:
        value: Value to sanitize
        max_depth: Max nesting depth
        max_string_length: Max string length
        max_dict_keys: Max keys per dict
        max_list_len: Max list length

    Returns:
        Sanitized value

    """
    return _sanitize_value(
        value,
        0,
        max_depth=max_depth,
        max_string_length=max_string_length,
        max_dict_keys=max_dict_keys,
        max_list_len=max_list_len,
    )
