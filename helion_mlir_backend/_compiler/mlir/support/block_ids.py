"""Helpers for resolving Helion block-id naming conventions."""

from __future__ import annotations


def block_id_from_key(value: object) -> int | None:
    """Return a block id from a ``block_size_N`` key."""
    if not isinstance(value, str) or not value.startswith("block_size_"):
        return None
    suffix = value.removeprefix("block_size_")
    return int(suffix) if suffix.isdigit() else None


def block_id_from_symbol(value: object) -> int | None:
    """Return a block id from a symbolic ``uN`` name."""
    if not isinstance(value, str) or not value.startswith("u"):
        return None
    suffix = value[1:]
    return int(suffix) if suffix.isdigit() else None
