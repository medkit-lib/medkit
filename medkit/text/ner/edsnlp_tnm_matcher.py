"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[edsnlp]`.
"""
from __future__ import annotations

__all__ = ["EDSNLPTNMMatcher"]

from typing import Iterator

import spacy

from medkit.core.text import Entity, Segment, span_utils
from medkit.core.text.operation import NEROperation
from medkit.text.ner.tnm_attribute import TNMAttribute
from medkit.text.spacy.edsnlp import build_tnm_attribute


class EDSNLPTNMMatcher(NEROperation):
    """TNM (Tumour/Node/Metastasis) string matcher based on `EDS-NPL's tnm pipeline
    <https://aphp.github.io/edsnlp/latest/pipelines/ner/tnm/>`.

    For each TNM string that is found, an entity will be created with an
    :class:`~medkit.text.ner.TNMAttribute` attribute attached to it containing
    normalized values of the TNM components.
    """

    def __init__(
        self,
        output_label: str = "TNM",
        attrs_to_copy: list[str] | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        output_label : str, default="TNM"
            Label to use for TNM entities created (the label of the
            attributes will always be "TNM")
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the input segment
            to the created TNM entity. Useful for propagating context attributes
            (negation, antecedent, etc).
        uid : str, optional
            Identifier of the matcher
        """
        super().__init__(output_label=output_label, attrs_to_copy=attrs_to_copy, uid=uid)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.output_label = output_label
        self.attrs_to_copy = attrs_to_copy

        self._edsnlp = spacy.blank("eds")
        self._edsnlp.add_pipe("eds.tnm")

    def run(self, segments: list[Segment]) -> list[Entity]:
        """Find and return TNM entities for all `segments`

        Parameters
        ----------
        segments : list of Segment
            List of segments into which to look for TNM strings

        Returns
        -------
        list of Entity
            TNM entities found in `segments`, with
            :class:`~medkit.text.ner.TNMAttribute` attributes
        """
        spacy_docs = self._edsnlp.pipe(s.text for s in segments)
        return [
            tnm_entity
            for segment, spacy_doc in zip(segments, spacy_docs)
            for tnm_entity in self._find_tnms_in_segment(segment, spacy_doc)
        ]

    def _find_tnms_in_segment(self, segment, spacy_doc) -> Iterator[Entity]:
        for spacy_span in spacy_doc.ents:
            # only TNM entities should be found
            assert spacy_span.label_ == "tnm"

            # convert span span start/end to medkit spans relative to doc
            text, spans = span_utils.extract(
                segment.text,
                segment.spans,
                [(spacy_span.start_char, spacy_span.end_char)],
            )

            # create attribute storing normalized TNM values
            # (only TNM attributes should be found)
            attr = build_tnm_attribute(spacy_span=spacy_span, spacy_label="tnm")
            assert isinstance(attr, TNMAttribute)

            # create entity
            entity = Entity(label=self.output_label, spans=spans, text=text, attrs=[attr])

            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])
                self._prov_tracer.add_prov(attr, self.description, source_data_items=[segment])

            # copy attrs from source segment to TNM entity
            for label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=label):
                    copied_attr = attr.copy()
                    entity.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            yield entity
