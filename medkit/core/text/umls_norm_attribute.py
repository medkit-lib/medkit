from __future__ import annotations

__all__ = ["UMLSNormAttribute"]

import dataclasses
from typing import Any, Dict, List, Optional

from typing_extensions import Self

from medkit.core import dict_conv
from medkit.core.text import EntityNormAttribute


@dataclasses.dataclass(init=False)
class UMLSNormAttribute(EntityNormAttribute):
    """
    Normalization attribute linking an entity to a CUI in the UMLS knowledge base

    Attributes
    ----------
    uid:
        Identifier of the attribute
    label:
        The attribute label, always set to :attr:`EntityNormAttribute.LABEL
        <.core.text.EntityNormAttribute.LABEL>`
    value:
        CUI prefixed with "umls:" (ex: "umls:C0011849")
    kb_name:
        Name of the knowledge base. Always "umls"
    kb_id:
        CUI (Concept Unique Identifier) to which the annotation should be linked
    cui:
        Convenience alias of `kb_id`
    kb_version:
        Version of the UMLS database (ex: "202AB")
    umls_version:
        Convenience alias of `kb_version`
    term:
        Optional normalized version of the entity text
    score:
        Optional score reflecting confidence of this link
    sem_types:
        Optional IDs of semantic types of the CUI (ex: ["T047"])
    metadata:
        Metadata of the attribute
    """

    sem_types: Optional[List[str]] = None

    def __init__(
        self,
        cui: str,
        umls_version: str,
        term: Optional[str] = None,
        score: Optional[float] = None,
        sem_types: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        uid: Optional[str] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        norm_dict = dict(
            uid=self.uid,
            cui=self.cui,
            umls_version=self.umls_version,
            term=self.term,
            score=self.score,
            sem_types=self.sem_types,
            metadata=self.metadata,
        )
        dict_conv.add_class_name_to_data_dict(self, norm_dict)
        return norm_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
            uid=data["uid"],
            cui=data["cui"],
            umls_version=data["umls_version"],
            term=data["term"],
            score=data["score"],
            sem_types=data["sem_types"],
            metadata=data["metadata"],
        )
