from __future__ import annotations

from typing import Iterable

__all__ = ["modules_are_available"]


def modules_are_available(modules: Iterable[str]) -> bool:
    from importlib.util import find_spec

    return all(find_spec(m) is not None for m in modules)
