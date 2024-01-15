from __future__ import annotations

__all__ = ["BratInputConverter", "BratOutputConverter"]

import logging
import re
from pathlib import Path
from typing import Any

from smart_open import open

import medkit.io._brat_utils as brat_utils
from medkit.core import (
    Attribute,
    InputConverter,
    OperationDescription,
    OutputConverter,
    ProvTracer,
    generate_id,
)
from medkit.core.text import (
    Entity,
    ModifiedSpan,
    Relation,
    Segment,
    Span,
    TextAnnotation,
    TextDocument,
    UMLSNormAttribute,
    span_utils,
    utils,
)
from medkit.io._brat_utils import (
    AttributeConf,
    BratAnnConfiguration,
    BratAttribute,
    BratEntity,
    BratNote,
    BratRelation,
    RelationConf,
)
from medkit.io._common import get_anns_by_type

TEXT_EXT = ".txt"
ANN_EXT = ".ann"
ANN_CONF_FILE = "annotation.conf"

logger = logging.getLogger(__name__)


_CUI_PATTERN = re.compile(r"\b[Cc]\d{7}\b")


class BratInputConverter(InputConverter):
    """Class in charge of converting brat annotations"""

    def __init__(
        self,
        detect_cuis_in_notes: bool = True,
        notes_label: str = "brat_note",
        uid: str | None = None,
    ):
        """Parameters
        ----------
        detect_cuis_in_notes : bool, default=True
            If `True`, strings looking like CUIs in annotator notes of entities
            will be converted to UMLS normalization attributes rather than creating
            an :class:`~.core.Attribute` with the whole note text as value.
        notes_label : str, default="brat_note",
            Label to use for attributes created from annotator notes.
        uid : str, optional
            Identifier of the converter.
        """
        if uid is None:
            uid = generate_id()

        self.notes_label = notes_label
        self.detect_cuis_in_notes = detect_cuis_in_notes
        self.uid = uid
        self._prov_tracer: ProvTracer | None = None

    @property
    def description(self) -> OperationDescription:
        return OperationDescription(
            uid=self.uid,
            name=self.__class__.__name__,
            class_name=self.__class__.__name__,
        )

    def set_prov_tracer(self, prov_tracer: ProvTracer):
        self._prov_tracer = prov_tracer

    def load(
        self,
        dir_path: str | Path,
        ann_ext: str = ANN_EXT,
        text_ext: str = TEXT_EXT,
    ) -> list[TextDocument]:
        """Create a list of TextDocuments from a folder containing text files
        and associated brat annotations files.

        Parameters
        ----------
        dir_path : str or Path
            The path to the directory containing the text files and the annotation
            files (.ann)
        ann_ext : str, optional
            The extension of the brat annotation file (e.g. .ann)
        text_ext : str, optional
            The extension of the text file (e.g. .txt)

        Returns
        -------
        list of TextDocument
            The list of TextDocuments
        """
        documents = []
        dir_path = Path(dir_path)

        # find all base paths with at least a corresponding text or ann file
        base_paths = set()
        for ann_path in sorted(dir_path.glob("*" + ann_ext)):
            base_paths.add(dir_path / ann_path.stem)
        for text_path in sorted(dir_path.glob("*" + text_ext)):
            base_paths.add(dir_path / text_path.stem)

        # load doc for each base_path
        for base_path in sorted(base_paths):
            text_path = base_path.with_suffix(text_ext)
            ann_path = base_path.with_suffix(ann_ext)

            if not text_path.exists():
                # ignore .ann without .txt
                logging.warning("Didn't find corresponding .txt for '%s', ignoring document", ann_path)
                continue

            if not ann_path.exists():
                # directly load .txt without .ann
                text = text_path.read_text(encoding="utf-8")
                metadata = {"path_to_text": str(text_path)}
                doc = TextDocument(text=text, metadata=metadata)
            else:
                # load both .txt and .ann
                doc = self.load_doc(ann_path=ann_path, text_path=text_path)
            documents.append(doc)

        if not documents:
            logger.warning("Didn't load any document from dir '%s'", dir_path)

        return documents

    def load_doc(self, ann_path: str | Path, text_path: str | Path) -> TextDocument:
        """Create a TextDocument from a .ann file and its associated .txt file

        Parameters
        ----------
        ann_path : str or Path
            The path to the brat annotation file.
        text_path : str or Path
            The path to the text document file.

        Returns
        -------
        TextDocument
            The document containing the text and the annotations
        """
        ann_path = Path(ann_path)
        text_path = Path(text_path)

        with open(text_path, encoding="utf-8") as fp:
            text = fp.read()

        anns = self.load_annotations(ann_path)

        metadata = {"path_to_text": str(text_path), "path_to_ann": str(ann_path)}

        doc = TextDocument(text=text, metadata=metadata)
        for ann in anns:
            doc.anns.add(ann)

        return doc

    def load_annotations(self, ann_file: str | Path) -> list[TextAnnotation]:
        """Load a .ann file and return a list of
        :class:`~medkit.core.text.annotation.Annotation` objects.

        Parameters
        ----------
        ann_file : str or Path
            Path to the .ann file.

        Returns
        -------
        list of TextAnnotation
            The list of text annotations
        """
        ann_file = Path(ann_file)

        brat_doc = brat_utils.parse_file(ann_file)
        anns_by_brat_id = {}

        # First convert entities, then relations, finally attributes
        # because new annotation identifier is needed
        for brat_entity in brat_doc.entities.values():
            # Rebuild spans, taking into account that Brat inserts a space
            # between each discontinuous span, and we need to account for it
            # with a ModifiedSpan
            spans = []
            for brat_span in brat_entity.span:
                if spans:
                    spans.append(ModifiedSpan(length=1, replaced_spans=[]))
                spans.append(Span(*brat_span))

            try:
                entity = Entity(
                    label=brat_entity.type,
                    spans=spans,
                    text=brat_entity.text,
                    metadata={"brat_id": brat_entity.uid},
                )
            except AssertionError as err:
                msg = f"Impossible to create an entity from '{ann_file.name}':{brat_entity.uid}."
                raise ValueError(msg) from err

            anns_by_brat_id[brat_entity.uid] = entity
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(entity, self.description, source_data_items=[])

        for brat_relation in brat_doc.relations.values():
            relation = Relation(
                label=brat_relation.type,
                source_id=anns_by_brat_id[brat_relation.subj].uid,
                target_id=anns_by_brat_id[brat_relation.obj].uid,
                metadata={"brat_id": brat_relation.uid},
            )
            anns_by_brat_id[brat_relation.uid] = relation
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(relation, self.description, source_data_items=[])

        for brat_attribute in brat_doc.attributes.values():
            attribute = Attribute(
                label=brat_attribute.type,
                value=brat_attribute.value,
                metadata={"brat_id": brat_attribute.uid},
            )
            anns_by_brat_id[brat_attribute.target].attrs.add(attribute)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(attribute, self.description, source_data_items=[])

        for brat_note in brat_doc.notes.values():
            # try to detect CUI in notes and recreate normalization attrs
            if self.detect_cuis_in_notes:
                cuis = _CUI_PATTERN.findall(brat_note.value)
                if cuis:
                    for cui in cuis:
                        attribute = UMLSNormAttribute(cui=cui, umls_version=None)
                        anns_by_brat_id[brat_note.target].attrs.add(attribute)
                        if self._prov_tracer is not None:
                            self._prov_tracer.add_prov(attribute, self.description, source_data_items=[])
                    continue

            # if no CUI detected, store note contents in plain attribute
            attribute = Attribute(label=self.notes_label, value=brat_note.value)
            anns_by_brat_id[brat_note.target].attrs.add(attribute)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(attribute, self.description, source_data_items=[])

        return list(anns_by_brat_id.values())


class BratOutputConverter(OutputConverter):
    """Class in charge of converting a list of TextDocuments into a
    brat collection file.

    .. hint::
        BRAT checks the coherence between span and text for each annotation.
        This converter adjusts the text and spans to get the right visualization
        and ensure compatibility.
    """

    def __init__(
        self,
        anns_labels: list[str] | None = None,
        attrs: list[str] | None = None,
        notes_label: str = "brat_note",
        ignore_segments: bool = True,
        convert_cuis_to_notes: bool = True,
        create_config: bool = True,
        top_values_by_attr: int = 50,
        uid: str | None = None,
    ):
        """Initialize the Brat output converter

        Parameters
        ----------
        anns_labels : list of str, optional
            Labels of medkit annotations to convert into Brat annotations.
            If `None` (default) all the annotations will be converted
        attrs : list of str, optional
            Labels of medkit attributes to add in the annotations that will be included.
            If `None` (default) all medkit attributes found in the segments or relations
            will be converted to Brat attributes
        notes_label : str, default="brat_note"
            Label of attributes that will be converted to annotator notes.
        ignore_segments : bool, default=True
            If `True` medkit segments will be ignored. Only entities, attributes and relations
            will be converted to Brat annotations.  If `False` the medkit segments will be
            converted to Brat annotations as well.
        convert_cuis_to_notes : bool, default=True
            If `True`, UMLS normalization attributes will be converted to
            annotator notes rather than attributes. For entities with multiple
            UMLS attributes, CUIs will be separated by spaces (ex: "C0011849 C0004096").
        create_config : bool, default=True
            Whether to create a configuration file for the generated collection.
            This file defines the types of annotations generated, it is necessary for the correct
            visualization on Brat.
        top_values_by_attr : int, default=50
            Defines the number of most common values by attribute to show in the configuration.
            This is useful when an attribute has a large number of values, only the 'top' ones
            will be in the config. By default, the top 50 of values by attr will be in the config.
        uid : str, optional
            Identifier of the converter
        """
        self.uid = uid or generate_id()
        self.anns_labels = anns_labels
        self.attrs = attrs
        self.notes_label = notes_label
        self.ignore_segments = ignore_segments
        self.convert_cuis_to_notes = convert_cuis_to_notes
        self.create_config = create_config
        self.top_values_by_attr = top_values_by_attr

    @property
    def description(self) -> OperationDescription:
        config = {
            "anns_labels": self.anns_labels,
            "attrs": self.attrs,
            "ignore_segments": self.ignore_segments,
            "create_config": self.create_config,
            "top_values_by_attr": self.top_values_by_attr,
        }
        return OperationDescription(uid=self.uid, class_name=self.__class__.__name__, config=config)

    def save(
        self,
        docs: list[TextDocument],
        dir_path: str | Path,
        doc_names: list[str] | None = None,
    ):
        """Convert and save a collection or list of TextDocuments into a Brat collection.
        For each collection or list of documents, a folder is created with '.txt' and '.ann'
        files; an 'annotation.conf' is saved if required.

        Parameters
        ----------
        docs : list of TextDocument
            List of medkit doc objects to convert
        dir_path : str or Path
            String or path object to save the generated files
        doc_names : list of str, optional
            Optional list with the names for the generated files. If 'None', 'uid' will
            be used as the name. Where 'uid.txt' has the raw text of the document and
            'uid.ann' the Brat annotation file.
        """
        if doc_names and len(doc_names) != len(docs):
            msg = "Size mismatch between names and docs"
            raise ValueError(msg)

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        config = BratAnnConfiguration(self.top_values_by_attr)

        for i, medkit_doc in enumerate(docs):
            text = medkit_doc.text
            doc_id = doc_names[i] if doc_names else medkit_doc.uid

            # convert medkit anns to brat format
            annotations = get_anns_by_type(medkit_doc, anns_labels=self.anns_labels)
            all_segments = annotations["entities"]

            if not self.ignore_segments:
                # In brat only entities exists, in some cases
                # a medkit document could include segments
                # that may be exported as entities
                all_segments += annotations["segments"]

            brat_anns = self._convert_medkit_anns_to_brat(
                segments=all_segments,
                relations=annotations["relations"],
                config=config,
                raw_text=text,
            )

            # save text file
            text_path = dir_path / f"{doc_id}{TEXT_EXT}"
            text_path.write_text(text, encoding="utf-8")
            # save ann file
            ann_path = dir_path / f"{doc_id}{ANN_EXT}"
            brat_str = "".join(f"{brat_ann.to_str()}" for brat_ann in brat_anns)
            ann_path.write_text(brat_str, encoding="utf-8")

        if self.create_config:
            # save configuration file by collection or list of documents
            conf_path = dir_path / ANN_CONF_FILE
            conf_path.write_text(config.to_str(), encoding="utf-8")

    def _convert_medkit_anns_to_brat(
        self,
        segments: list[Segment],
        relations: list[Relation],
        config: BratAnnConfiguration,
        raw_text: str,
    ) -> list[BratEntity | BratAttribute | BratRelation | BratNote]:
        """Convert Segments, Relations and Attributes into brat data structures

        Parameters
        ----------
        segments : list of Segment
            Medkit segments to convert
        relations : list of Relation
            Medkit relations to convert
        config : BratAnnConfiguration
            Optional `BratAnnConfiguration` structure, this object is updated
            with the types of the generated Brat annotations.
        raw_text : str
            Text of reference to get the original text of the annotations

        Returns
        -------
        list of BratEntity or BratAttribute or BratRelation or BratNote
            A list of brat annotations
        """
        nb_segment, nb_relation, nb_attribute, nb_note = 1, 1, 1, 1
        brat_entities_by_medkit_id = {}
        brat_anns = []

        # First convert segments then relations including its attributes
        for medkit_segment in segments:
            brat_entity = self._convert_segment_to_brat(medkit_segment, nb_segment, raw_text)
            brat_anns.append(brat_entity)
            # store link between medkit id and brat entities
            # (needed for relations)
            brat_entities_by_medkit_id[medkit_segment.uid] = brat_entity
            config.add_entity_type(brat_entity.type)
            nb_segment += 1

            # include selected attributes
            if self.attrs is None:
                attrs = medkit_segment.attrs.get()
            else:
                attrs = [a for label in self.attrs for a in medkit_segment.attrs.get(label=label)]
            for attr in attrs:
                # skip UMLS attributes that will be converted to notes
                if self.convert_cuis_to_notes and isinstance(attr, UMLSNormAttribute):
                    continue
                # skip attributes that will be converted to notes
                if attr.label == self.notes_label:
                    continue

                value = attr.to_brat()

                if isinstance(value, bool) and not value:
                    # in brat 'False' means the attributes does not exist
                    continue

                try:
                    brat_attr, attr_config = self._convert_attribute_to_brat(
                        label=attr.label,
                        value=value,
                        nb_attribute=nb_attribute,
                        target_brat_id=brat_entity.uid,
                        is_from_entity=True,
                    )
                    brat_anns.append(brat_attr)
                    config.add_attribute_type(attr_config)
                    nb_attribute += 1

                except TypeError as err:
                    logger.warning("Ignore attribute %s. %s", attr.uid, err)

            if self.convert_cuis_to_notes:
                cuis = [attr.kb_id for attr in attrs if isinstance(attr, UMLSNormAttribute)]
                if len(cuis):
                    brat_note = self._convert_umls_attributes_to_brat_note(
                        cuis=cuis,
                        nb_note=nb_note,
                        target_brat_id=brat_entity.uid,
                    )
                    brat_anns.append(brat_note)
                    nb_note += 1

            note_attrs = medkit_segment.attrs.get(label=self.notes_label)
            if note_attrs:
                values = [a.to_brat() for a in note_attrs]
                brat_note = self._convert_attributes_to_brat_note(
                    values=values,
                    nb_note=nb_note,
                    target_brat_id=brat_entity.uid,
                )
                brat_anns.append(brat_note)
                nb_note += 1

        for medkit_relation in relations:
            try:
                brat_relation, relation_config = self._convert_relation_to_brat(
                    medkit_relation, nb_relation, brat_entities_by_medkit_id
                )
                brat_anns.append(brat_relation)
                config.add_relation_type(relation_config)
                nb_relation += 1
            except ValueError as err:
                logger.warning("Ignore relation %s. %s", medkit_relation.uid, err)
                continue

            # Note: it seems that brat does not support attributes for relations
            # include selected attributes
            if self.attrs is None:
                attrs = medkit_relation.attrs.get()
            else:
                attrs = [a for label in self.attrs for a in medkit_relation.attrs.get(label=label)]
            for attr in attrs:
                value = attr.to_brat()

                if isinstance(value, bool) and not value:
                    continue

                try:
                    brat_attr, attr_config = self._convert_attribute_to_brat(
                        label=attr.label,
                        value=value,
                        nb_attribute=nb_attribute,
                        target_brat_id=brat_relation.uid,
                        is_from_entity=False,
                    )
                    brat_anns.append(brat_attr)
                    config.add_attribute_type(attr_config)
                    nb_attribute += 1
                except TypeError as err:
                    logger.warning("Ignore attribute %s. %s", attr.uid, err)

        return brat_anns

    @staticmethod
    def _ensure_text_and_spans(segment: Segment, raw_text: str) -> tuple[str, list[tuple[int, int]]]:
        """Ensure consistency between the segment and the raw text.
        The text of a BRAT annotation can't contain multiple white spaces (including a newline character).
        This method cleans the fragments' text and adjust its spans to point to the same location in the raw text.

        Parameters
        ----------
        segment : Segment
            Segment to ensure
        raw_text : str
            Text of reference

        Returns
        -------
        text : str
            The cleaned text
        spans : list of tuple
            The adjusted spans
        """
        pattern_to_clean = r"(\s*\n+\s*)"
        segment_spans = span_utils.normalize_spans(segment.spans)
        texts_brat, spans_brat = [], []

        for fragment in segment_spans:
            text = raw_text[fragment.start : fragment.end]
            offset = fragment.start
            # remove leading spaces from text or multiple spaces
            text_stripped, start_text, end_text = utils.strip(text, offset)
            real_offset = offset + start_text

            # create text and spans without blank regions
            for match in re.finditer(pattern_to_clean, text_stripped):
                end_fragment = start_text + match.start()
                texts_brat.append(raw_text[start_text:end_fragment])
                spans_brat.append((start_text, end_fragment))
                start_text = match.end() + real_offset

            # add last fragment
            texts_brat.append(raw_text[start_text:end_text])
            spans_brat.append((start_text, end_text))

        text_brat = " ".join(texts_brat)
        return text_brat, spans_brat

    def _convert_segment_to_brat(self, segment: Segment, nb_segment: int, raw_text: str) -> BratEntity:
        """Get a brat entity from a medkit segment

        Parameters
        ----------
        segment : Segment
            A medkit segment to convert into brat format
        nb_segment : int
            The current counter of brat segments
        raw_text : str
            Text of reference to get the original text of the segment

        Returns
        -------
        BratEntity
            The equivalent brat entity of the medkit segment
        """
        if nb_segment <= 0:
            msg = f"Number of segments {nb_segment} must be strictly positive"
            raise ValueError(msg)

        brat_id = f"T{nb_segment}"
        # brat does not support spaces in labels
        type_ = segment.label.replace(" ", "_")
        text, spans = self._ensure_text_and_spans(segment, raw_text)
        return BratEntity(brat_id, type_, spans, text)

    @staticmethod
    def _convert_relation_to_brat(
        relation: Relation,
        nb_relation: int,
        brat_entities_by_segment_id: dict[str, BratEntity],
    ) -> tuple[BratRelation, RelationConf]:
        """Get a brat relation from a medkit relation

        Parameters
        ----------
        relation : Relation
            A medkit relation to convert into brat format
        nb_relation : int
            The current counter of brat relations
        brat_entities_by_segment_id : dict of str to BratEntity
            A dict to map medkit ID to brat annotation

        Returns
        -------
        relation : BratRelation
            The equivalent brat relation of the medkit relation
        config : RelationConf
            Configuration of the brat attribute

        Raises
        ------
        ValueError
            When the source or target was not found in the mapping object
        """
        if nb_relation <= 0:
            msg = f"Number of relations {nb_relation} must be strictly positive"
            raise ValueError(msg)

        brat_id = f"R{nb_relation}"
        # brat does not support spaces in labels
        type_ = relation.label.replace(" ", "_")
        subj = brat_entities_by_segment_id.get(relation.source_id)
        obj = brat_entities_by_segment_id.get(relation.target_id)

        if subj is None or obj is None:
            msg = "Entity target/source was not found."
            raise ValueError(msg)

        relation_conf = RelationConf(type_, arg1=subj.type, arg2=obj.type)
        return BratRelation(brat_id, type_, subj.uid, obj.uid), relation_conf

    @staticmethod
    def _convert_attribute_to_brat(
        label: str,
        value: str | None,
        nb_attribute: int,
        target_brat_id: str,
        is_from_entity: bool,
    ) -> tuple[BratAttribute, AttributeConf]:
        """Get a brat attribute from a medkit attribute

        Parameters
        ----------
        label : str
            Attribute label to convert into brat format
        value : str, optional
            Attribute value
        nb_attribute : int
            The current counter of brat attributes
        target_brat_id : str
            Corresponding target brat ID

        Returns
        -------
        attribute : BratAttribute
            The equivalent brat attribute of the medkit attribute
        config : AttributeConf
            Configuration of the brat attribute
        """
        if nb_attribute <= 0:
            msg = f"Number of attributes {nb_attribute} must be strictly positive"
            raise ValueError(msg)

        brat_id = f"A{nb_attribute}"
        type_ = label.replace(" ", "_")

        value = brat_utils.ensure_attr_value(value)
        attr_conf = AttributeConf(from_entity=is_from_entity, type=type_, value=value)
        return BratAttribute(brat_id, type_, target_brat_id, value), attr_conf

    @staticmethod
    def _convert_umls_attributes_to_brat_note(
        cuis: list[str],
        nb_note: int,
        target_brat_id: str,
    ) -> BratNote:
        """Get a brat note from a medkit umls norm attribute

        Parameters
        ----------
        cuis : list of str
            CUI to convert to brat note
        nb_note : int
            The current counter of brat notes
        target_brat_id : str
            Corresponding target brat ID

        Returns
        -------
        BratNote
            The equivalent brat note of the medkit umls attribute
        """
        if nb_note <= 0:
            msg = f"Number of notes {nb_note} must be strictly positive"
            raise ValueError(msg)

        brat_id = f"#{nb_note}"
        return BratNote(uid=brat_id, target=target_brat_id, value=" ".join(cuis))

    @staticmethod
    def _convert_attributes_to_brat_note(
        values: list[Any],
        nb_note: int,
        target_brat_id: str,
    ) -> BratNote:
        """Get a brat note from medkit attribute values

        Parameters
        ----------
        values : list of Any
            Attribute values
        nb_note : int
            The current counter of brat notes
        target_brat_id : str
            Corresponding target brat ID

        Returns
        -------
        BratNote
            The equivalent brat note of the medkit attribute values
        """
        if nb_note <= 0:
            msg = f"Number of notes {nb_note} must be strictly positive"
            raise ValueError(msg)

        brat_id = f"#{nb_note}"
        value = "\n".join(str(v) for v in values if v is not None)
        return BratNote(uid=brat_id, target=target_brat_id, value=value)
