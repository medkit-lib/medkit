__all__ = ["ADICAPNormAttribute"]

import dataclasses
from typing import Any, Dict, Optional

from typing_extensions import Self

from medkit.core import dict_conv
from medkit.core.text import EntityNormAttribute


@dataclasses.dataclass
class ADICAPNormAttribute(EntityNormAttribute):
    """
    Attribute describing tissue sample using the ADICAP (Association pour le
    Développement de l'Informatique en Cytologie et Anatomo-Pathologie) coding.

    :see: https://smt.esante.gouv.fr/wp-json/ans/terminologies/document?terminologyId=terminologie-adicap&fileName=cgts_sem_adicap_fiche-detaillee.pdf

    This class is replicating EDS-NLP's `Adicap` class, making it a medkit
    `Attribute`.

    The `code` field fully describes the tissue sample. Additional information
    is derived from `code` in human readable fields (`sampling_code`,
    `technic`, `organ`, `pathology`, `pathology_type`, `behaviour_type`)

    Attributes
    ----------
    uid:
        Identifier of the attribute
    label:
        The attribute label, always set to :attr:`EntityNormAttribute.LABEL
        <.core.text.EntityNormAttribute.LABEL>`
    value:
        ADICAP code prefix with "adicap:" (ex: "adicap:BHGS0040")
    code:
        ADICAP code as a string (ex: "BHGS0040")
    kb_id:
        Same as `code`
    sampling_mode:
        Sampling mode (ex: "BIOPSIE CHIRURGICALE")
    technic:
        Sampling technic (ex: "HISTOLOGIE ET CYTOLOGIE PAR INCLUSION")
    organ:
        Organ and regions (ex: "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)")
    pathology:
        General pathology (ex: "PATHOLOGIE GÉNÉRALE NON TUMORALE")
    pathology_type:
        Pathology type (ex: "ETAT SUBNORMAL - LESION MINEURE")
    behaviour_type:
        Behaviour type (ex: "CARACTERES GENERAUX")
    metadata:
        Metadata of the attribute
    """

    sampling_mode: Optional[str]
    technic: Optional[str]
    organ: Optional[str]
    pathology: Optional[str]
    pathology_type: Optional[str]
    behaviour_type: Optional[str]

    def __init__(
        self,
        code: str,
        sampling_mode: Optional[str] = None,
        technic: Optional[str] = None,
        organ: Optional[str] = None,
        pathology: Optional[str] = None,
        pathology_type: Optional[str] = None,
        behaviour_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        uid: Optional[str] = None,
    ):
        super().__init__(kb_name="adicap", kb_id=code, metadata=metadata, uid=uid)

        self.sampling_mode = sampling_mode
        self.technic = technic
        self.organ = organ
        self.pathology = pathology
        self.pathology_type = pathology_type
        self.behaviour_type = behaviour_type

    @property
    def code(self) -> str:
        return self.kb_id

    def to_dict(self) -> Dict[str, Any]:
        adicap_dict = dict(
            uid=self.uid,
            code=self.code,
            sampling_mode=self.sampling_mode,
            technic=self.technic,
            organ=self.organ,
            pathology=self.pathology,
            pathology_type=self.pathology_type,
            behaviour_type=self.behaviour_type,
            metadata=self.metadata,
        )
        dict_conv.add_class_name_to_data_dict(adicap_dict, self)
        return adicap_dict

    @classmethod
    def from_dict(cls, adicap_dict: Dict[str, Any]) -> Self:
        return cls(
            uid=adicap_dict["uid"],
            code=adicap_dict["code"],
            sampling_mode=adicap_dict["sampling_mode"],
            technic=adicap_dict["technic"],
            organ=adicap_dict["organ"],
            pathology=adicap_dict["pathology"],
            pathology_type=adicap_dict["pathology_type"],
            behaviour_type=adicap_dict["behaviour_type"],
            metadata=adicap_dict["metadata"],
        )
