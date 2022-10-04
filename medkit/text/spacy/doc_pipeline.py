__all__ = ["SpacyDocPipeline"]
import warnings
from typing import List, Optional, Union

from spacy import Language

from medkit.core import Collection, DocOperation
from medkit.core.text import TextDocument
from medkit.text.spacy import spacy_utils


class SpacyDocPipeline(DocOperation):
    """DocPipeline to obtain annotations created using spacy"""

    def __init__(
        self,
        nlp: Language,
        medkit_labels_anns: Optional[List[str]] = None,
        medkit_attrs: Optional[List[str]] = None,
        spacy_entities: Optional[List[str]] = None,
        spacy_span_groups: Optional[List[str]] = None,
        spacy_attrs: Optional[List[str]] = None,
        op_id: Optional[str] = None,
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
            Name of span extensions to convert into medkit attributes.
            If `None` (default) all non-None extensions will be added for each annotation with
            a medkit ID.
        op_id:
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

    def run(self, medkit_docs: Union[List[TextDocument], Collection]) -> None:
        """Run a spacy pipeline on a list of medkit documents.
        Each medkit document is converted to spacy document (Doc object),
        with the selected annotations and attributes. Then, the spacy pipeline
        is executed and finally, the new annotations and attributes are
        converted into medkit annotations.

        Parameters
        ----------
        medkit_docs:
            List or collection of TextDocuments on which to run the pipeline
        """
        if isinstance(medkit_docs, Collection):
            medkit_docs = [
                medkit_doc
                for medkit_doc in medkit_docs.documents
                if isinstance(medkit_doc, TextDocument)
            ]

        for medkit_doc in medkit_docs:
            if medkit_doc.text is None:
                warnings.warn(
                    f"The document with id {medkit_doc.id} has no text, it is not"
                    " converted"
                )
                continue

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
                rebuild_medkit_anns_and_attrs=False,
            )
            # annotate
            # add new annotations
            for ann in anns:
                medkit_doc.add_annotation(ann)
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(
                        ann,
                        self.description,
                        source_data_items=[raw_segment],
                    )

            # add new attributes in each annotation
            for ann_id, attrs in attrs_by_ann_id.items():
                ann = medkit_doc.get_annotation_by_id(ann_id)
                for attr in attrs:
                    ann.add_attr(attr)
                    if self._prov_tracer is not None:
                        # if ann is an existing annotation, in terms
                        # of provenance, the annotation was used to
                        # generate the attribute, else, it was regenerate using
                        # raw_text_segment
                        source_data_item = raw_segment if ann in anns else ann
                        self._prov_tracer.add_prov(
                            attr, self.description, source_data_items=[source_data_item]
                        )