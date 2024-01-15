from __future__ import annotations

__all__ = ["OperationDescription"]

import dataclasses
from typing import Any


@dataclasses.dataclass
class OperationDescription:
    """Description of a specific instance of an operation

    Attributes
    ----------
    uid : str
        The unique identifier of the instance described
    name : str
        The name of the operation. Can be the same as `class_name` or something
        more specific, for operations with a behavior that can be customized
        (for instance a rule-based entity matcher with user-provided rules, or a
        model-based entity matcher with a user-provided model)
    class_name : str, optional
        The name of the class of the operation
    config : dict of str to Any, optional
        The specific configuration of the instance
    """

    uid: str
    name: str
    class_name: str | None = None
    config: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"uid": self.uid, "name": self.name, "class_name": self.class_name, "config": self.config}
