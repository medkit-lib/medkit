from __future__ import annotations

__all__ = ["EntityNormAttribute"]

import dataclasses
from typing import Any, ClassVar

from typing_extensions import Self

from medkit.core import dict_conv
from medkit.core.attribute import Attribute


@dataclasses.dataclass(init=False)
class EntityNormAttribute(Attribute):
    """Normalization attribute linking an entity to an ID in a knowledge base

    Attributes
    ----------
    uid : str
        Identifier of the attribute
    label : str
        The attribute label, always set to :attr:`EntityNormAttribute.LABEL
        <.core.text.EntityNormAttribute.LABEL>`
    value : Any
        String representation of the normalization, containing `kb_id`, along
        with `kb_name` if available (ex: "umls:C0011849"). For special cases
        where only `term` is available, it is used as value.
    kb_name : str, optional
        Name of the knowledge base (ex: "icd"). Should always be provided except
        in special cases when we just want to store a normalized term.
    kb_id : Any, optional
        ID in the knowledge base to which the annotation should be linked.
        Should always be provided except in special cases when we just want to
        store a normalized term.
    kb_version : str, optional
        Optional version of the knowledge base.
    term : str, optional
        Optional normalized version of the entity text.
    score : float, optional
        Optional score reflecting confidence of this link.
    metadata : dict of str to Any
        Metadata of the attribute
    """

    kb_name: str | None
    kb_id: Any | None
    kb_version: str | None
    term: str | None
    score: float | None

    LABEL: ClassVar[str] = "NORMALIZATION"
    """
    Label used for all normalization attributes
    """

    def __init__(
        self,
        kb_name: str | None,
        kb_id: Any | None,
        kb_version: str | None = None,
        term: str | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        if kb_id is None and term is None:
            msg = "Must provide at least kb_id or term"
            raise ValueError(msg)

        if kb_name and kb_id:
            value = f"{kb_name}:{kb_id}"
        elif kb_id:
            value = str(kb_id)
        else:
            value = term

        super().__init__(label=self.LABEL, value=value, metadata=metadata, uid=uid)

        self.kb_name = kb_name
        self.kb_id = kb_id
        self.kb_version = kb_version
        self.term = term
        self.score = score

    def to_brat(self) -> str:
        return self.value

    def to_spacy(self) -> str:
        return self.value

    def to_dict(self) -> dict[str, Any]:
        norm_dict = {
            "uid": self.uid,
            "label": self.label,
            "kb_name": self.kb_name,
            "kb_id": self.kb_id,
            "kb_version": self.kb_version,
            "term": self.term,
            "score": self.score,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, norm_dict)
        return norm_dict

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> Self:
        return cls(
            uid=data_dict["uid"],
            kb_name=data_dict["kb_name"],
            kb_id=data_dict["kb_id"],
            kb_version=data_dict["kb_version"],
            term=data_dict["term"],
            score=data_dict["score"],
            metadata=data_dict["metadata"],
        )
