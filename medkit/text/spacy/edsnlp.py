"""
This package needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit[edsnlp]`.
"""

__all__ = [
    "EDSNLPPipeline",
    "EDSNLPDocPipeline",
    "build_date_attribute",
    "build_adicap_attribute",
    "build_tnm_attribute",
    "build_measurement_attribute",
    "build_value_attribute",
    "build_score_attribute",
    "build_context_attribute",
    "build_history_attribute",
    "DEFAULT_ATTRIBUTE_FACTORIES",
]

from typing import Callable, Dict, Optional, List

from edsnlp.pipelines.ner.adicap.models import AdicapCode as EDSNLP_AdicapCode
from edsnlp.pipelines.ner.tnm.model import TNM as EDSNLP_TNM
from edsnlp.pipelines.misc.dates.models import (
    AbsoluteDate as EDSNLP_AbsoluteDate,
    RelativeDate as EDSNLP_RelativeDate,
    Duration as EDSNLP_Duration,
    Direction as EDSNLP_Direction,
)
from edsnlp.pipelines.misc.measurements.measurements import (
    SimpleMeasurement as EDSNLP_SimpleMeasurement,
)
from spacy import Language
from spacy.tokens import Span as SpacySpan
from spacy.tokens.underscore import Underscore

from medkit.core import Attribute
from medkit.text.ner import (
    ADICAPNormAttribute,
    DateAttribute,
    RelativeDateAttribute,
    RelativeDateDirection,
    DurationAttribute,
)
from medkit.text.ner.tnm_attribute import TNMAttribute
from medkit.text.spacy import SpacyPipeline, SpacyDocPipeline


def build_date_attribute(spacy_span: SpacySpan, spacy_label: str) -> Attribute:
    """
    Build a medkit date attribute from an EDS-NLP attribute with a date object
    as value.

    Parameters
    ----------
    spacy_span
        Spacy span having an ESD-NLP date attribute
    spacy_label
        Label of the date attribute on `spacy_spacy`. Ex: "date", "consultation_date"

    Returns
    -------
    Attribute
        :class:`~medkit.text.ner.DateAttribute`,
        :class:`~medkit.text.ner.RelativeDateAttribute` or
        :class:`~medkit.text.ner.DurationAttribute` instance, depending on the
        EDS-NLP attribute
    """

    value = spacy_span._.get(spacy_label)
    if isinstance(value, EDSNLP_AbsoluteDate):
        return DateAttribute(
            label=spacy_label,
            year=value.year,
            month=value.month,
            day=value.day,
            hour=value.hour,
            minute=value.minute,
            second=value.second,
        )
    elif isinstance(value, EDSNLP_RelativeDate):
        direction = (
            RelativeDateDirection.PAST
            if value.direction is EDSNLP_Direction.PAST
            else RelativeDateDirection.FUTURE
        )
        return RelativeDateAttribute(
            label=spacy_label,
            direction=direction,
            years=value.year,
            months=value.month,
            weeks=value.week,
            days=value.day,
            hours=value.hour,
            minutes=value.minute,
            seconds=value.second,
        )
    elif isinstance(value, EDSNLP_Duration):
        return DurationAttribute(
            label=spacy_label,
            years=value.year,
            months=value.month,
            weeks=value.week,
            days=value.day,
            hours=value.hour,
            minutes=value.minute,
            seconds=value.second,
        )
    else:
        raise ValueError(
            f"Unexpected value type: {type(value)} for spaCy attribute with label"
            f" '{spacy_label}'"
        )


def build_adicap_attribute(
    spacy_span: SpacySpan, spacy_label: str
) -> ADICAPNormAttribute:
    """
    Build a medkit ADICAP normalization attribute from an EDS-NLP attribute with
    an ADICAP object as value.

    Parameters
    ----------
    spacy_span
        Spacy span having an ADICAP object as value
    spacy_label
        Label of the attribute on `spacy_spacy`. Ex: "adicap"

    Returns
    -------
    ADICAPNormAttribute
        Medkit ADICAP normalization attribute
    """

    value = spacy_span._.get(spacy_label)
    assert isinstance(value, EDSNLP_AdicapCode)
    return ADICAPNormAttribute(
        code=value.code,
        sampling_mode=value.sampling_mode,
        technic=value.technic,
        organ=value.organ,
        pathology=value.pathology,
        pathology_type=value.pathology_type,
        behaviour_type=value.behaviour_type,
    )


def build_tnm_attribute(spacy_span: SpacySpan, spacy_label: str) -> TNMAttribute:
    """
    Build a medkit TNM attribute from an EDS-NLP attribute with a TNM object as
    value.

    Parameters
    ----------
    spacy_span
        Spacy span having a TNM object as value
    spacy_label
        Label of the attribute on `spacy_spacy`. Ex: "tnm"

    Returns
    -------
    TNMAttribute
        Medkit TNM attribute
    """

    value = spacy_span._.get(spacy_label)
    assert isinstance(value, EDSNLP_TNM)
    return TNMAttribute(
        prefix=value.prefix,
        tumour=value.tumour,
        tumour_specification=value.tumour_specification,
        node=value.node,
        node_specification=value.node_specification,
        node_suffix=value.node_suffix,
        metastasis=value.metastasis,
        resection_completeness=value.resection_completeness,
        version=value.version,
        version_year=value.version_year,
    )


def build_measurement_attribute(spacy_span: SpacySpan, spacy_label: str) -> Attribute:
    """
    Build a medkit attribute from an EDS-NLP attribute with a measurement object
    as value.

    Parameters
    ----------
    spacy_span
        Spacy span having a measurement object as value
    spacy_label
        Label of the attribute on `spacy_spacy`. Ex: "size", "weight", "bmi"

    Returns
    -------
    Attribute
        Medkit attribute with normalized measurement value and "unit" metadata
    """
    value = spacy_span._.get(spacy_label)
    assert isinstance(value, EDSNLP_SimpleMeasurement)
    return Attribute(
        label=spacy_label, value=value.value, metadata={"unit": value.unit}
    )


def build_value_attribute(spacy_span: SpacySpan, spacy_label: str) -> Attribute:
    """
    Build a medkit attribute from an EDS-NLP "value" attribute with a custom object as value:

    - if the value is an EDS-NLP `Adipcap` object, a
      :class:`~medkit.text.ner.ADICAPNormAttribute` instance is returned;
    - if the value is an EDS-NLP `TNN` object, a
      :class:`~medkit.text.ner.tnm_attribute.TNMAttribute` instance is returned;
    - if the value is an EDS-NLP `SimpleMeasurement` object, a
      :class:`~medkit.core.Attribute` instance is returned.


    Otherwise an error is raised.

    Parameters
    ----------
    spacy_span
        Spacy span having an attribute custom object as value
    spacy_label
        Label of the attribute on `spacy_spacy`. Ex: "value"

    Returns
    -------
    Attribute
        Medkit attribute corresponding to the spacy attribute value
    """

    value = spacy_span._.get(spacy_label)
    if isinstance(value, EDSNLP_AdicapCode):
        return build_adicap_attribute(spacy_span, spacy_label)
    elif isinstance(value, EDSNLP_TNM):
        return build_tnm_attribute(spacy_span, spacy_label)
    elif isinstance(value, EDSNLP_SimpleMeasurement):
        return build_measurement_attribute(spacy_span, spacy_label)
    else:
        raise ValueError(
            f"Unexpected value type: {type(value)} for spaCy attribute with label"
            f" '{spacy_label}'"
        )


def build_score_attribute(spacy_span: SpacySpan, spacy_label: str) -> Attribute:
    """
    Build a medkit attribute from an EDS-NLP "score_name" and corresponding
    "score_value" attribute.

    Parameters
    ----------
    spacy_span
        Spacy span having "score_name" and "score_value" attributes
    spacy_label
        Must be "score_name"

    Returns
    -------
    Attribute
        Medkit attribute with "score_name" value as label and "score_value" value as
        value
    """

    assert spacy_label == "score_name"
    label = spacy_span._.score_name
    value = spacy_span._.score_value
    method = spacy_span._.get("score_method")
    metadata = {"method": method} if method is not None else None
    return Attribute(label=label, value=value, metadata=metadata)


def build_context_attribute(spacy_span: SpacySpan, spacy_label: str) -> Attribute:
    """
    Build a medkit attribute from an EDS-NLP context/qualifying attribute, adding the
    cues as metadata

    Parameters
    ----------
    spacy_span
        Spacy span having a context/qualifying attribute
    spacy_label
        Label of the attribute on `spacy_spacy`. Ex: "negation", "hypothesis", etc

    Returns
    -------
    Attribute
        Medkit attribute with optional "cues" metadata
    """

    value = spacy_span._.get(spacy_label)
    cues = spacy_span._.get(f"{spacy_label}_cues")
    metadata = {"cues": [c.text for c in cues]} if cues else None
    return Attribute(label=spacy_label, value=value, metadata=metadata)


def build_history_attribute(spacy_span: SpacySpan, spacy_label: str) -> Attribute:
    """
    Build a medkit attribute from an EDS-NLP "history" attribute, adding the cues as
    metadata

    Parameters
    ----------
    spacy_span
        Spacy span having a "history" attribute
    spacy_label
        Must be "history"

    Returns
    -------
    Attribute
        Medkit attribute corresponding to the spacy attribute
    """

    assert spacy_label == "history"
    value = spacy_span._.history
    history_cues = spacy_span._.get("history_cues")
    recent_cues = spacy_span._.get("recent_cues")
    metadata = {}
    if history_cues is not None:
        metadata["history_cues"] = [c.text for c in history_cues]
    if recent_cues is not None:
        metadata["recent_cues"] = [c.text for c in recent_cues]
    return Attribute(label="history", value=value, metadata=metadata)


DEFAULT_ATTRIBUTE_FACTORIES = {
    # created by several components
    "value": build_value_attribute,
    # from eds.dates
    "date": build_date_attribute,
    # from eds.consultation_dates
    "consultation_date": build_date_attribute,
    # from eds.score and some subclasses
    "score_name": build_score_attribute,
    # from eds.family
    "family": build_context_attribute,
    # from eds.hypothesis
    "hypothesis": build_context_attribute,
    # from eds.negation
    "negation": build_context_attribute,
    # from eds.reported_speech
    "reported_speech": build_context_attribute,
    # from eds.history
    "history": build_history_attribute,
}
"""Pre-defined attribute factories to handle EDS-NLP attributes"""

_ATTR_LABELS_TO_IGNORE = {
    # text after spaCy pre-preprocessing
    "normalized_variant",
    # should be in metadata of entities matched by eds.contextual-matcher but we don't support that
    "assigned",
    "source",
    # declared but unused attribute of eds.dates
    "datetime",
    # unsupported experimental feature of eds.dates
    "period"
    # ignored because each entity matched by eds.reason will also have its own is_reason attribute
    "ents_reason",
    # redundant with value attr
    "adicap",
    # will be set as value of score_name attr
    "score_value",
    # added to metadata of score_name attr
    "score_method",
    # context/qualifying attributes with deprecated aliases and cues included in metadata
    "family_",
    "family_cues",
    "history_",
    "history_cues",
    "recent_cues",
    "antecedents",
    "antecedents_",
    "antecedents_cues",
    "antecedent",
    "antecedent_",
    "antecedent_cues",
    "hypothesis_",
    "hypothesis_cues",
    "negation_",
    "negated",
    "polarity_",
    "negation_cues",
    "reported_speech_",
    "reported_speech_cues",
}


class EDSNLPPipeline(SpacyPipeline):
    """Segment annotator relying on an EDS-NLP pipeline"""

    def __init__(
        self,
        nlp: Language,
        spacy_entities: Optional[List[str]] = None,
        spacy_span_groups: Optional[List[str]] = None,
        spacy_attrs: Optional[List[str]] = None,
        medkit_attribute_factories: Optional[
            Dict[str, Callable[[SpacySpan, str], Attribute]]
        ] = None,
        name: Optional[str] = None,
        uid: Optional[str] = None,
    ):
        """Initialize the segment annotator

        Parameters
        ----------
        nlp:
            Language object with the loaded pipeline from Spacy
        spacy_entities:
            Labels of new spacy entities (`doc.ents`) to convert into medkit entities.
            If `None` (default) all the new spacy entities will be converted
        spacy_span_groups:
            Name of new spacy span groups (`doc.spans`) to convert into medkit segments.
            If `None` (default) new spacy span groups will be converted
        spacy_attrs:
            Name of span extensions to convert into medkit attributes. If
            `None`, all non-redundant EDS-NLP attributes will be handled.
        medkit_attribute_factories:
            Mapping of factories in charge of converting spacy attributes to
            medkit attributes. Factories will receive a spacy span and an an
            attribute label when called. The key in the mapping is the attribute
            label.
            Pre-defined default factories are listed in
            :const:`~DEFAULT_ATTRIBUTE_FACTORIES`
        name:
            Name describing the pipeline (defaults to the class name).
        uid:
            Identifier of the pipeline
        """

        if medkit_attribute_factories is None:
            medkit_attribute_factories = DEFAULT_ATTRIBUTE_FACTORIES
        else:
            medkit_attribute_factories = {
                **DEFAULT_ATTRIBUTE_FACTORIES,
                **medkit_attribute_factories,
            }

        if spacy_attrs is None:
            # default to all span attributes except blacklisted ones
            spacy_attrs = [
                attr
                for attr in Underscore.span_extensions
                if attr not in _ATTR_LABELS_TO_IGNORE
            ]

        super().__init__(
            nlp=nlp,
            spacy_entities=spacy_entities,
            spacy_span_groups=spacy_span_groups,
            spacy_attrs=spacy_attrs,
            medkit_attribute_factories=medkit_attribute_factories,
            name=name,
            uid=uid,
        )


class EDSNLPDocPipeline(SpacyDocPipeline):
    """
    DocPipeline to obtain annotations created using EDS-NLP
    """

    def __init__(
        self,
        nlp: Language,
        medkit_labels_anns: Optional[List[str]] = None,
        medkit_attrs: Optional[List[str]] = None,
        spacy_entities: Optional[List[str]] = None,
        spacy_span_groups: Optional[List[str]] = None,
        spacy_attrs: Optional[List[str]] = None,
        medkit_attribute_factories: Optional[
            Dict[str, Callable[[SpacySpan, str], Attribute]]
        ] = None,
        name: Optional[str] = None,
        uid: Optional[str] = None,
    ):
        """Initialize the pipeline

        Parameters
        ----------
        nlp:
            Language object with the loaded pipeline from Spacy
        medkit_labels_anns:
            Labels of medkit annotations to include in the spacy document.
            If `None` (default) all the annotations will be included.
        medkit_attrs:
            Labels of medkit attributes to add in the annotations that will be included.
            If `None` (default) all the attributes will be added as `custom attributes`
            in each annotation included.
        spacy_entities:
            Labels of new spacy entities (`doc.ents`) to convert into medkit entities.
            If `None` (default) all the new spacy entities will be converted and added into
            its origin medkit document.
        spacy_span_groups:
            Name of new spacy span groups (`doc.spans`) to convert into medkit segments.
            If `None` (default) new spacy span groups will be converted and added into
            its origin medkit document.
        spacy_attrs:
            Name of span extensions to convert into medkit attributes. If
            `None`, all non-redundant EDS-NLP attributes will be handled.
        medkit_attribute_factories:
            Mapping of factories in charge of converting spacy attributes to
            medkit attributes. Factories will receive a spacy span and an an
            attribute label when called. The key in the mapping is the attribute
            label.
            Pre-defined default factories are listed in
            :const:`~DEFAULT_ATTRIBUTE_FACTORIES`
        name:
            Name describing the pipeline (defaults to the class name).
        uid:
            Identifier of the pipeline
        """

        # use pre-defined attribute factory
        if medkit_attribute_factories is None:
            medkit_attribute_factories = DEFAULT_ATTRIBUTE_FACTORIES
        else:
            medkit_attribute_factories = {
                **DEFAULT_ATTRIBUTE_FACTORIES,
                **medkit_attribute_factories,
            }

        if spacy_attrs is None:
            # default to all span attributes except blacklisted ones
            spacy_attrs = [
                attr
                for attr in Underscore.span_extensions
                if attr not in _ATTR_LABELS_TO_IGNORE
            ]

        super().__init__(
            nlp=nlp,
            medkit_labels_anns=medkit_labels_anns,
            medkit_attrs=medkit_attrs,
            spacy_entities=spacy_entities,
            spacy_span_groups=spacy_span_groups,
            spacy_attrs=spacy_attrs,
            medkit_attribute_factories=medkit_attribute_factories,
            name=name,
            uid=uid,
        )
