"""This package needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit[edsnlp]`.
"""
from __future__ import annotations

__all__ = ["TNMAttribute"]

import dataclasses
from typing import Any, ClassVar

from edsnlp.pipelines.ner.tnm.model import (
    TNM,
    Metastasis,
    Node,
    Prefix,
    Specification,
    Tumour,
)
from typing_extensions import Self

from medkit.core import Attribute, dict_conv


@dataclasses.dataclass
class TNMAttribute(Attribute):
    """Attribute destructuring the fields of a TNM string.

    The TNM (Tumour/Node/Metastasis) system is used to describe cancer stages.

    This class is replicating EDS-NLP's `TDM` class, making it a medkit
    :class:`Attribute`.

    Attributes
    ----------
    uid:
        Identifier of the attribute
    label:
        The attribute label, always set to :attr:`TNMAttribute.LABEL`
    value:
        Normalized string representation of the TNM (ex: "pTxN1M1")
    tumour:
        Tumour score
    tumour_specification:
        Tumour specification
    tumour_suffix:
        Tumour suffix
    node:
        Node score
    node_specification:
        Node specification
    node_suffix:
        Node suffix
    metastasis:
        Metastasis score
    resection_completeness:
        Resection completeness (R factor)
    version:
        Version (ex: "uicc", "accj")
    version_year:
        Version year
    """

    prefix: Prefix | None
    tumour: Tumour | None
    tumour_specification: Specification | None
    tumour_suffix: str | None
    node: Node | None
    node_specification: Specification | None
    node_suffix: str | None
    metastasis: Metastasis | None
    resection_completeness: int | None
    version: str | None
    version_year: int | None

    LABEL: ClassVar[str] = "TNM"
    """
    Label used for all TNM attributes
    """

    def __init__(
        self,
        prefix: Prefix | None = None,
        tumour: Tumour | None = None,
        tumour_specification: Specification | None = None,
        tumour_suffix: str | None = None,
        node: Node | None = None,
        node_specification: Specification | None = None,
        node_suffix: str | None = None,
        metastasis: Metastasis | None = None,
        resection_completeness: int | None = None,
        version: str | None = None,
        version_year: int | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        # use EDS-NLP's TNM class to build string representation
        value = TNM(
            prefix=prefix,
            tumour=tumour,
            tumour_specification=tumour_specification,
            tumour_suffix=tumour_suffix,
            node=node,
            node_specification=node_specification,
            node_suffix=node_suffix,
            metastasis=metastasis,
            resection_completeness=resection_completeness,
            version=version,
            version_year=version_year,
        ).norm()

        super().__init__(label=self.LABEL, value=value, metadata=metadata, uid=uid)

        self.prefix = prefix
        self.tumour = tumour
        self.tumour_specification = tumour_specification
        self.tumour_suffix = tumour_suffix
        self.node = node
        self.node_specification = node_specification
        self.node_suffix = node_suffix
        self.metastasis = metastasis
        self.resection_completeness = resection_completeness
        self.version = version
        self.version_year = version_year

    def to_brat(self) -> str:
        return self.value

    def to_spacy(self) -> str:
        return self.value

    def to_dict(self) -> dict[str, Any]:
        tnm_dict = {
            "uid": self.uid,
            "prefix": self.prefix,
            "tumour": self.tumour,
            "tumour_suffix": self.tumour_suffix,
            "tumour_specification": self.tumour_specification,
            "node": self.node,
            "node_specification": self.node_specification,
            "node_suffix": self.node_suffix,
            "metastasis": self.metastasis,
            "resection_completeness": self.resection_completeness,
            "version": self.version,
            "version_year": self.version_year,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, tnm_dict)
        return tnm_dict

    @classmethod
    def from_dict(cls, tnm_dict: dict[str, Any]) -> Self:
        return cls(
            uid=tnm_dict["uid"],
            prefix=tnm_dict["prefix"],
            tumour=tnm_dict["tumour"],
            tumour_suffix=tnm_dict["tumour_suffix"],
            tumour_specification=tnm_dict["tumour_specification"],
            node=tnm_dict["node"],
            node_specification=tnm_dict["node_specification"],
            node_suffix=tnm_dict["node_suffix"],
            metastasis=tnm_dict["metastasis"],
            resection_completeness=tnm_dict["resection_completeness"],
            version=tnm_dict["version"],
            version_year=tnm_dict["version_year"],
            metadata=tnm_dict["metadata"],
        )
