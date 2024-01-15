"""This package needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit[edsnlp]`.
"""
from __future__ import annotations

__all__ = [
    "EDSNLPPipeline",
    "EDSNLPDocPipeline",
    "build_date_attribute",
    "build_duration_attribute",
    "build_adicap_attribute",
    "build_tnm_attribute",
    "build_measurement_attribute",
    "DEFAULT_ATTRIBUTE_FACTORIES",
]

from typing import TYPE_CHECKING, Callable

from edsnlp.pipelines.misc.dates.models import AbsoluteDate as EDSNLP_AbsoluteDate
from edsnlp.pipelines.misc.dates.models import Direction as EDSNLP_Direction
from edsnlp.pipelines.misc.dates.models import Duration as EDSNLP_Duration
from edsnlp.pipelines.misc.dates.models import RelativeDate as EDSNLP_RelativeDate
from edsnlp.pipelines.misc.measurements.measurements import (
    SimpleMeasurement as EDSNLP_SimpleMeasurement,
)
from edsnlp.pipelines.ner.adicap.models import AdicapCode as EDSNLP_AdicapCode
from edsnlp.pipelines.ner.tnm.model import TNM as EDSNLP_TNM
from spacy.tokens.underscore import Underscore

from medkit.core import Attribute
from medkit.text.ner import (
    ADICAPNormAttribute,
    DateAttribute,
    DurationAttribute,
    RelativeDateAttribute,
    RelativeDateDirection,
)
from medkit.text.ner.tnm_attribute import TNMAttribute
from medkit.text.spacy import SpacyDocPipeline, SpacyPipeline

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Span as SpacySpan


def build_date_attribute(spacy_span: SpacySpan, spacy_label: str) -> Attribute:
    """Build a medkit date attribute from an EDS-NLP attribute with a date object
    as value.

    Parameters
    ----------
    spacy_span : SpacySpan
        Spacy span having an ESD-NLP date attribute
    spacy_label : str
        Label of the date attribute on `spacy_spacy`. Ex: "date", "consultation_date"

    Returns
    -------
    Attribute
        :class:`~medkit.text.ner.DateAttribute` or
        :class:`~medkit.text.ner.RelativeDateAttribute` instance, depending on
        the EDS-NLP attribute
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
    elif isinstance(value, EDSNLP_RelativeDate):  # noqa: RET505
        direction = (
            RelativeDateDirection.PAST if value.direction is EDSNLP_Direction.PAST else RelativeDateDirection.FUTURE
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
    else:
        msg = f"Unexpected value type: {type(value)} for spaCy attribute with label '{spacy_label}'"
        raise TypeError(msg)


def build_duration_attribute(spacy_span: SpacySpan, spacy_label: str) -> DurationAttribute:
    """Build a medkit duration attribute from an EDS-NLP attribute with a duration
    object as value.

    Parameters
    ----------
    spacy_span : SpacySpan
        Spacy span having an ESD-NLP date attribute
    spacy_label : str
        Label of the date attribute on `spacy_spacy`. Ex: "duration"

    Returns
    -------
    DurationAttribute
        Medkit duration attribute
    """
    value = spacy_span._.get(spacy_label)
    assert isinstance(value, EDSNLP_Duration)
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


def build_adicap_attribute(spacy_span: SpacySpan, spacy_label: str) -> ADICAPNormAttribute:
    """Build a medkit ADICAP normalization attribute from an EDS-NLP attribute with
    an ADICAP object as value.

    Parameters
    ----------
    spacy_span : SpacySpan
        Spacy span having an ADICAP object as value
    spacy_label : str
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
    """Build a medkit TNM attribute from an EDS-NLP attribute with a TNM object as
    value.

    Parameters
    ----------
    spacy_span : SpacySpan
        Spacy span having a TNM object as value
    spacy_label : str
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
    """Build a medkit attribute from an EDS-NLP attribute with a measurement object
    as value.

    Parameters
    ----------
    spacy_span : SpacySpan
        Spacy span having a measurement object as value
    spacy_label : str
        Label of the attribute on `spacy_spacy`. Ex: "size", "weight", "bmi"

    Returns
    -------
    Attribute
        Medkit attribute with normalized measurement value and "unit" metadata
    """
    value = spacy_span._.get(spacy_label)
    assert isinstance(value, EDSNLP_SimpleMeasurement)
    return Attribute(label=spacy_label, value=value.value, metadata={"unit": value.unit})


DEFAULT_ATTRIBUTE_FACTORIES = {
    # from eds.adicap
    "adicap": build_adicap_attribute,
    # from eds.tnm
    "tnm": build_tnm_attribute,
    # from eds.dates
    "date": build_date_attribute,
    "duration": build_duration_attribute,
    # from eds.consultation_dates
    "consultation_date": build_date_attribute,
    # from eds.measurements
    "weight": build_measurement_attribute,
    "size": build_measurement_attribute,
    "bmi": build_measurement_attribute,
    "volume": build_measurement_attribute,
}
"""Pre-defined attribute factories to handle EDS-NLP attributes"""

_ATTR_LABELS_TO_IGNORE = {
    # seems to always have an identical attr with a more specific label
    # since EDSNLP 0.9
    "value",
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
    # redundant with score attr with more specific label
    "score_value",
    # could be in metadata of score attrs but not worth the trouble
    "score_method",
    # context/qualifying attributes with deprecated aliases
    # cues could be included in metadata but not worth the trouble
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
        spacy_entities: list[str] | None = None,
        spacy_span_groups: list[str] | None = None,
        spacy_attrs: list[str] | None = None,
        medkit_attribute_factories: dict[str, Callable[[SpacySpan, str], Attribute]] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Initialize the segment annotator

        Parameters
        ----------
        nlp : Language
            Language object with the loaded pipeline from Spacy
        spacy_entities : list of str, optional
            Labels of new spacy entities (`doc.ents`) to convert into medkit entities.
            If `None` (default) all the new spacy entities will be converted
        spacy_span_groups : list of str, optional
            Name of new spacy span groups (`doc.spans`) to convert into medkit segments.
            If `None` (default) new spacy span groups will be converted
        spacy_attrs : list of str, optional
            Name of span extensions to convert into medkit attributes. If
            `None`, all non-redundant EDS-NLP attributes will be handled.
        medkit_attribute_factories : dict of str to Callable, optional
            Mapping of factories in charge of converting spacy attributes to
            medkit attributes. Factories will receive a spacy span and an an
            attribute label when called. The key in the mapping is the attribute
            label.
            Pre-defined default factories are listed in
            :const:`~DEFAULT_ATTRIBUTE_FACTORIES`
        name : str, optional
            Name describing the pipeline (defaults to the class name).
        uid : str, optional
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
            spacy_attrs = [attr for attr in Underscore.span_extensions if attr not in _ATTR_LABELS_TO_IGNORE]

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
    """DocPipeline to obtain annotations created using EDS-NLP"""

    def __init__(
        self,
        nlp: Language,
        medkit_labels_anns: list[str] | None = None,
        medkit_attrs: list[str] | None = None,
        spacy_entities: list[str] | None = None,
        spacy_span_groups: list[str] | None = None,
        spacy_attrs: list[str] | None = None,
        medkit_attribute_factories: dict[str, Callable[[SpacySpan, str], Attribute]] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Initialize the pipeline

        Parameters
        ----------
        nlp : Language
            Language object with the loaded pipeline from Spacy
        medkit_labels_anns : list of str, optional
            Labels of medkit annotations to include in the spacy document.
            If `None` (default) all the annotations will be included.
        medkit_attrs : list of str, optional
            Labels of medkit attributes to add in the annotations that will be included.
            If `None` (default) all the attributes will be added as `custom attributes`
            in each annotation included.
        spacy_entities : list of str, optional
            Labels of new spacy entities (`doc.ents`) to convert into medkit entities.
            If `None` (default) all the new spacy entities will be converted and added into
            its origin medkit document.
        spacy_span_groups : list of str, optional
            Name of new spacy span groups (`doc.spans`) to convert into medkit segments.
            If `None` (default) new spacy span groups will be converted and added into
            its origin medkit document.
        spacy_attrs : list of str, optional
            Name of span extensions to convert into medkit attributes. If
            `None`, all non-redundant EDS-NLP attributes will be handled.
        medkit_attribute_factories : dict of str to Callable, optional
            Mapping of factories in charge of converting spacy attributes to
            medkit attributes. Factories will receive a spacy span and an an
            attribute label when called. The key in the mapping is the attribute
            label.
            Pre-defined default factories are listed in
            :const:`~DEFAULT_ATTRIBUTE_FACTORIES`
        name : str, optional
            Name describing the pipeline (defaults to the class name).
        uid : str, optional
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
            spacy_attrs = [attr for attr in Underscore.span_extensions if attr not in _ATTR_LABELS_TO_IGNORE]

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
