from __future__ import annotations

__all__ = ["batch_iter", "batch_list", "modules_are_available"]

import importlib.util
from typing import Any, Iterator


def batch_iter(values: Iterator[Any], batch_size: int) -> Iterator[list[Any]]:
    """Group values yielded by an iterator into batches.

    Parameters
    ----------
    values : iterator of Any
        The iterator yielding values to batch.
    batch_size : int
        Length of batches (the last batch may be smaller).

    Returns
    -------
    iterator of list of Any
        Iterator yielding lists of `batch_size` items (the last list yielded may
        be smaller).
    """
    while True:
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(values))  # noqa: PERF401
        except StopIteration:
            yield batch
            return
        yield batch


def batch_list(values: list[Any], batch_size: int) -> Iterator[list[Any]]:
    """Split list into smaller batches.

    Parameters
    ----------
    values : list of Any
        The list containing values to batch.
    batch_size : int
        Length of batches (the last batch may be smaller).

    Returns
    -------
    iterator of list of Any
        Iterator yielding lists of `batch_size` items (the last list yielded may
        be smaller).
    """
    for i in range(0, len(values), batch_size):
        yield list[i : i + batch_size]


def modules_are_available(modules: list[str]):
    return all(importlib.util.find_spec(m) is not None for m in modules)
