"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[spacy]`.
"""
from __future__ import annotations

__all__ = ["SpacyInputConverter", "SpacyOutputConverter"]


from typing import TYPE_CHECKING

from medkit.core import OperationDescription, ProvTracer, generate_id
from medkit.core.text import TextDocument
from medkit.text.spacy.spacy_utils import (
    build_spacy_doc_from_medkit_doc,
    extract_anns_and_attrs_from_spacy_doc,
)

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Doc


class SpacyInputConverter:
    """Class in charge of converting spacy documents into a collection of TextDocuments."""

    def __init__(
        self,
        entities: list[str] | None = None,
        span_groups: list[str] | None = None,
        attrs: list[str] | None = None,
        uid: str | None = None,
    ):
        """Initialize the spacy input converter

        Parameters
        ----------
        entities : list of str, optional
            Labels of spacy entities (`doc.ents`) to convert into medkit entities.
            If `None` (default) all spacy entities will be converted and added into
            its origin medkit document.
        span_groups : list of str, optional
            Name of groups of spacy spans (`doc.spans`) to convert into medkit segments.
            If `None` (default) all groups of spacy spans will be converted and added into
            the medkit document.
        attrs : list of str, optional
            Name of span extensions to convert into medkit attributes.
            If `None` (default) all non-None extensions will be added for each annotation
        uid : str, optional
            Identifier of the converter
        """
        if uid is None:
            uid = generate_id()

        self.uid = uid
        self._prov_tracer: ProvTracer | None = None

        self.entities = entities
        self.span_groups = span_groups
        self.attrs = attrs

    @property
    def description(self) -> OperationDescription:
        config = {
            "entities": self.entities,
            "span_groups": self.span_groups,
            "attrs": self.attrs,
        }

        return OperationDescription(
            uid=self.uid,
            name=self.__class__.__name__,
            class_name=self.__class__.__name__,
            config=config,
        )

    def set_prov_tracer(self, prov_tracer: ProvTracer):
        self._prov_tracer = prov_tracer

    def load(self, spacy_docs: list[Doc]) -> list[TextDocument]:
        """Create a list of TextDocuments from a list of spacy Doc objects.
        Depending on the configuration of the converted, the selected annotations
        and attributes are included in the documents.

        Parameters
        ----------
        spacy_docs : list of Doc
            A list of spacy documents to convert

        Returns
        -------
         list of TextDocument
            A list of TextDocuments
        """
        medkit_docs = []
        for spacy_doc in spacy_docs:
            # create a new medkit document (TextDocument object)
            medkit_doc = TextDocument(text=spacy_doc.text_with_ws)
            anns = self._load_anns(spacy_doc)
            for ann in anns:
                medkit_doc.anns.add(ann)
            medkit_docs.append(medkit_doc)

        return medkit_docs

    def _load_anns(self, spacy_doc: Doc):
        annotations, attributes_by_ann = extract_anns_and_attrs_from_spacy_doc(
            spacy_doc=spacy_doc,
            medkit_source_ann=None,
            entities=self.entities,
            span_groups=self.span_groups,
            attrs=self.attrs,
            rebuild_medkit_anns_and_attrs=True,
        )

        # add annotations
        for ann in annotations:
            if self._prov_tracer is not None:
                # the input converter does not know the source data item
                self._prov_tracer.add_prov(ann, self.description, source_data_items=[])

            if ann.uid in attributes_by_ann:
                attrs = attributes_by_ann[ann.uid]
                for attr in attrs:
                    ann.attrs.add(attr)
                    if self._prov_tracer is not None:
                        # the input converter does not know the source data item
                        self._prov_tracer.add_prov(attr, self.description, source_data_items=[])
        return annotations


class SpacyOutputConverter:
    """Class in charge of converting a list of TextDocuments into a
    list of spacy documents
    """

    def __init__(
        self,
        nlp: Language,
        apply_nlp_spacy: bool = False,
        labels_anns: list[str] | None = None,
        attrs: list[str] | None = None,
        uid: str | None = None,
    ):
        """Initialize the spacy output converter

        Parameters
        ----------
        nlp : Language
            Language object with the loaded pipeline from Spacy
        apply_nlp_spacy : bool, default=False
            If True, each component of `nlp` pipeline is applied to the new spacy document.
            Some features, such as 'POS TAG', are added by a component of the pipeline, this
            parameter should be True, in order to add such attributes.
            If False, the `nlp` pipeline is not applied in the spacy document, so the document
            contains only the annotations and attributes transferred by medkit.
        labels_anns : list of str, optional
            Labels of medkit annotations to include in the spacy document.
            If `None` (default) all the annotations will be included.
        attrs : list of str, optional
            Labels of medkit attributes to add in the annotations that will be included.
            If `None` (default) all the attributes will be added as `custom attributes`
            in each annotation included.
        uid : str, optional
            Identifier of the pipeline

        """
        if uid is None:
            uid = generate_id()

        self.uid = uid
        self._prov_tracer: ProvTracer | None = None

        self.nlp = nlp
        self.labels_anns = labels_anns
        self.attrs = attrs
        self.apply_nlp_spacy = apply_nlp_spacy

    @property
    def description(self) -> OperationDescription:
        # medkit does not support serialisation of nlp objects,
        # however version information like model name, author etc. is stored
        config = {
            "nlp_metadata": self.nlp.meta,
            "labels_anns": self.labels_anns,
            "attrs": self.attrs,
            "apply_nlp_spacy": self.apply_nlp_spacy,
        }
        return OperationDescription(uid=self.uid, class_name=self.__class__.__name__, config=config)

    def convert(self, medkit_docs: list[TextDocument]) -> list[Doc]:
        """Convert a list of TextDocuments into a list of spacy Doc objects.
        Depending on the configuration of the converted, the selected annotations
        and attributes are included in the documents.

        Parameters
        ----------
        medkit_docs : list of TextDocument
            A list of TextDocuments to convert

        Returns
        -------
        list of Doc
            A list of spacy Doc objects
        """
        spacy_docs = []
        for medkit_doc in medkit_docs:
            # create a spacy document from medkit with the selected annotations
            spacy_doc = build_spacy_doc_from_medkit_doc(
                nlp=self.nlp,
                medkit_doc=medkit_doc,
                labels_anns=self.labels_anns,
                attrs=self.attrs,
                include_medkit_info=False,
            )
            # each component of nlp spacy is applied
            if self.apply_nlp_spacy:
                spacy_doc = self.nlp(spacy_doc)

            spacy_docs.append(spacy_doc)

        return spacy_docs
