from __future__ import annotations

__all__ = ["SpacyDocPipeline"]

from typing import TYPE_CHECKING, Callable

from medkit.core import Attribute, DocOperation
from medkit.text.spacy import spacy_utils

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Span as SpacySpan

    from medkit.core.text import TextDocument


class SpacyDocPipeline(DocOperation):
    """DocPipeline to obtain annotations created using spacy"""

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
        self.medkit_labels_anns = medkit_labels_anns
        self.medkit_attrs = medkit_attrs
        self.spacy_entities = spacy_entities
        self.spacy_span_groups = spacy_span_groups
        self.spacy_attrs = spacy_attrs
        self.medkit_attribute_factories = medkit_attribute_factories

    def run(self, medkit_docs: list[TextDocument]) -> None:
        """Run a spacy pipeline on a list of medkit documents.
        Each medkit document is converted to spacy document (Doc object),
        with the selected annotations and attributes. Then, the spacy pipeline
        is executed and finally, the new annotations and attributes are
        converted into medkit annotations.

        Parameters
        ----------
        medkit_docs : list of TextDocument
            List of TextDocuments on which to run the pipeline
        """
        for medkit_doc in medkit_docs:
            # build spacy doc
            spacy_doc = spacy_utils.build_spacy_doc_from_medkit_doc(
                nlp=self.nlp,
                medkit_doc=medkit_doc,
                labels_anns=self.medkit_labels_anns,
                attrs=self.medkit_attrs,
                include_medkit_info=True,
            )
            # apply nlp spacy
            spacy_doc = self.nlp(spacy_doc)

            # get new annotations and attributes
            raw_segment = medkit_doc.raw_segment

            anns, attrs_by_ann_id = spacy_utils.extract_anns_and_attrs_from_spacy_doc(
                spacy_doc=spacy_doc,
                medkit_source_ann=raw_segment,
                entities=self.spacy_entities,
                span_groups=self.spacy_span_groups,
                attrs=self.spacy_attrs,
                attribute_factories=self.medkit_attribute_factories,
                rebuild_medkit_anns_and_attrs=False,
            )
            # annotate
            # add new annotations
            for ann in anns:
                medkit_doc.anns.add(ann)
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(
                        ann,
                        self.description,
                        source_data_items=[raw_segment],
                    )

            # add new attributes in each annotation
            for ann_id, attrs in attrs_by_ann_id.items():
                ann = medkit_doc.anns.get_by_id(ann_id)
                for attr in attrs:
                    ann.attrs.add(attr)
                    if self._prov_tracer is not None:
                        # if ann is an existing annotation, in terms
                        # of provenance, the annotation was used to
                        # generate the attribute, else, it was regenerate using
                        # raw_text_segment
                        source_data_item = raw_segment if ann in anns else ann
                        self._prov_tracer.add_prov(attr, self.description, source_data_items=[source_data_item])
