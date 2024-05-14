from __future__ import annotations

import importlib
import inspect
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types


__all__ = ["import_optional"]


def import_optional(name: str, extra: str | None = None) -> types.ModuleType:
    """Import an optional dependency or raise an appropriate error message.

    Parameters
    ----------
    name : str
        Module name to import.
    extra : str, optional
        Group of optional dependencies to suggest installing if the import fails.
        If unspecified, assume the extra is named after the caller's module.

    Returns
    -------
    ModuleType
        The successfully imported module.

    Raises
    ------
    ModuleNotFoundError
        In case the requested import failed.
    """
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as err:
        if not extra:
            calling_module = inspect.getmodulename(inspect.stack()[1][1])
            extra = calling_module.replace("_", "-") if calling_module else None

        note = f"Consider installing the appropriate extra with:\npip install 'medkit-lib[{extra}]'" if extra else None

        if sys.version_info >= (3, 11):
            if note:
                err.add_note(note)
            raise

        message = "\n".join([str(err), note or ""])
        raise ModuleNotFoundError(message) from err
    return module
