from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 9):
    from collections.abc import Iterable, Iterator
else:
    from typing import Iterable, Iterator


__all__ = ["batched"]


def _batched(iterable: Iterable[Any], n: int) -> Iterator[tuple[Any, ...]]:
    """Batch data from an iterable into n-sized tuples.

    The last tuple may be shorter than the requested batch size.

    Examples
    --------
    >>> list(_batched("ABCDEFG", 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    """
    from itertools import islice

    if n < 1:
        msg = "batch size must be at least one"
        raise ValueError(msg)

    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


if sys.version_info >= (3, 12):
    from itertools import batched
else:
    batched = _batched
