from __future__ import annotations

__all__ = ["modules_are_available"]


def modules_are_available(modules: list[str]):
    from importlib.util import find_spec

    return all(find_spec(m) is not None for m in modules)
