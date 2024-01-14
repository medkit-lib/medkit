from __future__ import annotations

__all__ = ["SpacyPipeline"]

from typing import TYPE_CHECKING, Callable

from medkit.core.operation import Operation
from medkit.text.spacy import spacy_utils

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Doc
    from spacy.tokens import Span as SpacySpan

    from medkit.core import Attribute
    from medkit.core.text import Segment


class SpacyPipeline(Operation):
    """Segment annotator relying on a Spacy pipeline"""

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
            Name of span extensions to convert into medkit attributes.
            If `None` (default) all non-None extensions will be added for each annotation with
            a medkit ID.
        medkit_attribute_factories : dict of str to Callable, optional
            Mapping of factories in charge of converting spacy attributes to
            medkit attributes. Factories will receive a spacy span and an an
            attribute label when called. The key in the mapping is the attribute
            label.
        name : str, optional
            Name describing the pipeline (defaults to the class name).
        uid : str, optional
            Identifier of the pipeline
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.nlp = nlp
        self.spacy_entities = spacy_entities
        self.spacy_span_groups = spacy_span_groups
        self.spacy_attrs = spacy_attrs
        self.medkit_attribute_factories = medkit_attribute_factories

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Run a spacy pipeline on a list of segments provided as input
        and returns a new list of segments.
        Each segment is converted to spacy document (Doc object).
        Then, the spacy pipeline is executed and finally, the new
        annotations and attributes are converted into medkit annotations.

        Parameters
        ----------
        segments : list of Segment
            List of segments on which to run the spacy pipeline

        Returns
        -------
        list of Segment
            List of new annotations
        """
        output_segments = []
        for segment in segments:
            # build spacy doc
            # TODO: transfer of annotations and attributes attached to
            # a segment are not currently supported, no anns are included
            spacy_doc = spacy_utils.build_spacy_doc_from_medkit_segment(
                nlp=self.nlp,
                segment=segment,
                annotations=[],
                attrs=[],
                include_medkit_info=True,
            )
            # apply nlp spacy
            spacy_doc = self.nlp(spacy_doc)

            new_segments = self._find_segments_in_spacy_doc(spacy_doc=spacy_doc, medkit_source_ann=segment)
            output_segments.extend(new_segments)

        return output_segments

    def _find_segments_in_spacy_doc(self, spacy_doc: Doc, medkit_source_ann: Segment):
        # get new annotations and attributes
        segments, attrs_by_ann_id = spacy_utils.extract_anns_and_attrs_from_spacy_doc(
            spacy_doc=spacy_doc,
            medkit_source_ann=medkit_source_ann,
            entities=self.spacy_entities,
            span_groups=self.spacy_span_groups,
            attrs=self.spacy_attrs,
            attribute_factories=self.medkit_attribute_factories,
            rebuild_medkit_anns_and_attrs=False,
        )
        for new_segment in segments:
            # add provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    new_segment,
                    self.description,
                    source_data_items=[medkit_source_ann],
                )

            # add attributes
            if new_segment.uid in attrs_by_ann_id:
                for attr in attrs_by_ann_id[new_segment.uid]:
                    new_segment.attrs.add(attr)
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(
                            attr,
                            self.description,
                            source_data_items=[medkit_source_ann],
                        )

            yield new_segment
