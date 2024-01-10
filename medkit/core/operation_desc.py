__all__ = ["OperationDescription"]

import dataclasses
from typing import Any, Dict, Optional


@dataclasses.dataclass
class OperationDescription:
    """Description of a specific instance of an operation

    Parameters
    ----------
    uid:
        The unique identifier of the instance described
    name:
        The name of the operation. Can be the same as `class_name` or something
        more specific, for operations with a behavior that can be customized
        (for instance a rule-based entity matcher with user-provided rules, or a
        model-based entity matcher with a user-provided model)
    class_name:
        The name of the class of the operation
    config:
        The specific configuration of the instance
    """

    uid: str
    name: str
    class_name: Optional[str] = None
    config: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(uid=self.uid, name=self.name, class_name=self.class_name, config=self.config)
