from __future__ import annotations

__all__ = ["UMLSNormAttribute"]

import dataclasses
from typing import Any

from typing_extensions import Self

from medkit.core import dict_conv
from medkit.core.text import EntityNormAttribute


@dataclasses.dataclass(init=False)
class UMLSNormAttribute(EntityNormAttribute):
    """Normalization attribute linking an entity to a CUI in the UMLS knowledge base

    Attributes
    ----------
    uid : str
        Identifier of the attribute
    label : str
        The attribute label, always set to :attr:`EntityNormAttribute.LABEL
        <.core.text.EntityNormAttribute.LABEL>`
    value : Any
        CUI prefixed with "umls:" (ex: "umls:C0011849")
    kb_name : str, optional
        Name of the knowledge base. Always "umls"
    kb_id : Any, optional
        CUI (Concept Unique Identifier) to which the annotation should be linked
    cui : str
        Convenience alias of `kb_id`
    kb_version : str, optional
        Version of the UMLS database (ex: "202AB")
    umls_version : str
        Convenience alias of `kb_version`
    term : str, optional
        Optional normalized version of the entity text
    score : float, optional
        Optional score reflecting confidence of this link
    sem_types : list of str, optional
        Optional IDs of semantic types of the CUI (ex: ["T047"])
    metadata : dict of str to Any
        Metadata of the attribute
    """

    sem_types: list[str] | None = None

    def __init__(
        self,
        cui: str,
        umls_version: str,
        term: str | None = None,
        score: float | None = None,
        sem_types: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        super().__init__(
            kb_name="umls",
            kb_id=cui,
            kb_version=umls_version,
            term=term,
            score=score,
            metadata=metadata,
            uid=uid,
        )
        self.sem_types = sem_types

    @property
    def cui(self):
        return self.kb_id

    @property
    def umls_version(self):
        return self.kb_version

    def to_dict(self) -> dict[str, Any]:
        norm_dict = {
            "uid": self.uid,
            "cui": self.cui,
            "umls_version": self.umls_version,
            "term": self.term,
            "score": self.score,
            "sem_types": self.sem_types,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, norm_dict)
        return norm_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            uid=data["uid"],
            cui=data["cui"],
            umls_version=data["umls_version"],
            term=data["term"],
            score=data["score"],
            sem_types=data["sem_types"],
            metadata=data["metadata"],
        )
