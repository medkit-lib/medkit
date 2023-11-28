__all__ = [
    "DoccanoTask",
    "DoccanoClientConfig",
    "DoccanoInputConverter",
    "DoccanoOutputConverter",
]

import dataclasses
import enum
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self
from zipfile import ZipFile

from medkit.core import Attribute, OperationDescription, ProvTracer
from medkit.core.id import generate_id, generate_deterministic_id
from medkit.core.text import Entity, Relation, Span, TextDocument, span_utils

from medkit.io._common import get_anns_by_type

logger = logging.getLogger(__name__)


class DoccanoTask(enum.Enum):
    """Supported doccano tasks. The task defines
    the type of document to convert.

    Attributes
    ----------
    TEXT_CLASSIFICATION
        Documents with a category
    RELATION_EXTRACTION
        Documents with entities and relations (including IDs)
    SEQUENCE_LABELING
        Documents with entities in tuples
    """

    TEXT_CLASSIFICATION = "text_classification"
    RELATION_EXTRACTION = "relation_extraction"
    SEQUENCE_LABELING = "sequence_labeling"


@dataclasses.dataclass
class DoccanoClientConfig:
    """A class representing the configuration in the doccano client.
    The default values are the default values used by doccano.

    Attributes
    ----------
    column_text:
        Name or key representing the text
    column_label:
        Name or key representing the label
    """

    column_text: str = "text"
    column_label: str = "label"


# FIXME: datamodels to factorize in _doccano_utils
@dataclasses.dataclass()
class _DoccanoEntity:
    id: int
    start_offset: int
    end_offset: int
    label: str

    def to_dict(self) -> Dict[str, Any]:
        entity_dict = dict(
            id=self.id,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            label=self.label,
        )
        return entity_dict


@dataclasses.dataclass()
class _DoccanoEntityTuple:
    start_offset: int
    end_offset: int
    label: str

    def to_tuple(self) -> Tuple[int, int, str]:
        return (self.start_offset, self.end_offset, self.label)


@dataclasses.dataclass()
class _DoccanoRelation:
    id: int
    from_id: int
    to_id: int
    type: str

    def to_dict(self) -> Dict[str, Any]:
        relation_dict = dict(
            id=self.id,
            from_id=self.from_id,
            to_id=self.to_id,
            type=self.type,
        )
        return relation_dict


@dataclasses.dataclass()
class _DoccanoDocRelationExtraction:
    text: str
    entities: List[_DoccanoEntity]
    relations: List[_DoccanoRelation]
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(
        cls, doc_line: Dict[str, Any], client_config: DoccanoClientConfig
    ) -> Self:
        text: str = doc_line.pop(client_config.column_text)
        entities = [_DoccanoEntity(**ann) for ann in doc_line.pop("entities")]
        relations = [_DoccanoRelation(**ann) for ann in doc_line.pop("relations")]
        # in doccano, metadata is what remains after removing key fields
        metadata = doc_line
        return cls(text=text, entities=entities, relations=relations, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        doc_dict = dict(text=self.text)
        doc_dict["entities"] = [ent.to_dict() for ent in self.entities]
        doc_dict["relations"] = [rel.to_dict() for rel in self.relations]
        doc_dict.update(self.metadata)
        return doc_dict


@dataclasses.dataclass()
class _DoccanoDocSeqLabeling:
    text: str
    entities: List[_DoccanoEntityTuple]
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(
        cls, doc_line: Dict[str, Any], client_config: DoccanoClientConfig
    ) -> Self:
        text = doc_line.pop(client_config.column_text)
        entities = [
            _DoccanoEntityTuple(*ann)
            for ann in doc_line.pop(client_config.column_label)
        ]
        # in doccano, metadata is what remains after removing key fields
        metadata = doc_line
        return cls(text=text, entities=entities, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        doc_dict = dict(text=self.text)
        doc_dict["label"] = [ent.to_tuple() for ent in self.entities]
        doc_dict.update(self.metadata)
        return doc_dict


@dataclasses.dataclass()
class _DoccanoDocTextClassification:
    text: str
    label: str
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(
        cls, doc_line: Dict[str, Any], client_config: DoccanoClientConfig
    ) -> Self:
        text = doc_line.pop(client_config.column_text)
        label = doc_line.pop(client_config.column_label)[0]

        if not isinstance(label, str):
            raise TypeError(
                "The label must be a string. Please check if the document corresponds"
                " to a text classification task rather than sequence labeling"
            )
        # in doccano, metadata is what remains after removing key fields
        metadata = doc_line
        return cls(text=text, label=label, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        doc_dict = dict(text=self.text, label=[str(self.label)])
        doc_dict.update(self.metadata)
        return doc_dict


class DoccanoInputConverter:
    """Convert doccano files (.JSONL) containing annotations for a given task.

    For each line, a :class:`~.core.text.TextDocument` will be created.
    The doccano files can be loaded from a directory with zip files or from a jsonl file.

    The converter supports custom configuration to define the parameters used by doccano
    when importing the data (c.f. :class:`~.io.doccano.DoccanoClientConfig`)

    .. warning::
        If the option *Count grapheme clusters as one character*  was selected
        when creating the doccano project, the converted documents are
        likely to have alignment problems; the converter does not support this option.
    """

    def __init__(
        self,
        task: DoccanoTask,
        client_config: Optional[DoccanoClientConfig] = None,
        attr_label: str = "doccano_category",
        uid: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        task:
            The doccano task for the input converter
        client_config:
            Optional client configuration to define default values in doccano interface.
            This config can change, for example, the name of the text field or labels.
        attr_label:
            The label to use for the medkit attribute that represents the doccano category.
            This is related to :class:`~.io.DoccanoTask.TEXT_CLASSIFICATION` projects.
        uid:
            Identifier of the converter.
        """
        if uid is None:
            uid = generate_id()

        if client_config is None:
            client_config = DoccanoClientConfig()

        self.uid = uid
        self.client_config = client_config
        self.task = task
        self.attr_label = attr_label
        self._prov_tracer: Optional[ProvTracer] = None

    def set_prov_tracer(self, prov_tracer: ProvTracer):
        """Enable provenance tracing.

        Parameters
        ----------
        prov_tracer:
            The provenance tracer used to trace the provenance.
        """
        self._prov_tracer = prov_tracer

    @property
    def description(self) -> OperationDescription:
        """Contains all the input converter init parameters."""
        return OperationDescription(
            uid=self.uid,
            name=self.__class__.__name__,
            class_name=self.__class__.__name__,
            config=dict(task=self.task.value),
        )

    def load_from_directory_zip(self, dir_path: Union[str, Path]) -> List[TextDocument]:
        """Create a list of TextDocuments from zip files in a directory.
        The zip files should contain a JSONL file coming from doccano.

        Parameters
        ----------
        dir_path:
            The path to the directory containing zip files.

        Returns
        -------
        List[TextDocument]
            A list of TextDocuments
        """
        documents = []
        for path_zip in sorted(Path(dir_path).glob("*.zip")):
            documents.extend(self.load_from_zip(path_zip))

        if len(documents) == 0:
            logger.warning(f"No .zip nor .jsonl found in '{dir_path}'")

        return documents

    def load_from_zip(self, input_file: Union[str, Path]) -> List[TextDocument]:
        """
        Create a list of TextDocuments from a zip file containing a JSONL file
        coming from doccano.

        Parameters
        ----------
        input_file:
            The path to the zip file containing a docanno JSONL file

        Returns
        -------
        List[TextDocument]
            A list of TextDocuments
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(input_file, mode="r") as zip_file:
                filename = zip_file.namelist()[0]
                unzipped_file = Path(tmpdir) / filename
                zip_file.extract(filename, tmpdir)
            return self.load_from_file(unzipped_file)

    def load_from_file(self, input_file: Union[str, Path]) -> List[TextDocument]:
        """Create a list of TextDocuments from a doccano JSONL file.

        Parameters
        ----------
        input_file:
            The path to the JSONL file containing doccano annotations

        Returns
        -------
        List[TextDocument]
            A list of TextDocuments
        """
        documents = []
        with open(Path(input_file), encoding="utf-8") as fp:
            for line in fp:
                doc_line = json.loads(line)
                doc = self._parse_doc_line(doc_line)
                documents.append(doc)

        self._check_crlf_character(documents)
        return documents

    def _check_crlf_character(self, documents: List[TextDocument]):
        """Check if the list of converted documents contains the CRLF character.
        This character is the only indicator available to warn
        if there are alignment problems in the documents"""
        if (
            self.task == DoccanoTask.RELATION_EXTRACTION
            or self.task == DoccanoTask.SEQUENCE_LABELING
        ):
            nb_docs_with_warning = sum(
                document.text.find("\r\n") != -1 for document in documents
            )

            if nb_docs_with_warning > 0:
                logger.warning(
                    f"{nb_docs_with_warning}/{len(documents)} documents contain"
                    " '\\r\\n' characters. If you have selected 'Count grapheme"
                    " clusters as one character' when creating the doccano project,"
                    " converted documents are likely to have alignment problems.\n"
                    " Please ignore this message if you did not select this option when"
                    " creating the project."
                )

    def _parse_doc_line(self, doc_line: Dict[str, Any]) -> TextDocument:
        """Parse a doc_line into a TextDocument depending on the task

        Parameters
        ----------
        doc_line:
            A dictionary representing an annotation from doccano

        Returns
        -------
        TextDocument
            A document with parsed annotations.
        """
        if self.task == DoccanoTask.RELATION_EXTRACTION:
            return self._parse_doc_line_relation_extraction(doc_line=doc_line)
        if self.task == DoccanoTask.TEXT_CLASSIFICATION:
            return self._parse_doc_line_text_classification(doc_line=doc_line)
        if self.task == DoccanoTask.SEQUENCE_LABELING:
            return self._parse_doc_line_seq_labeling(doc_line=doc_line)

    def _parse_doc_line_relation_extraction(
        self, doc_line: Dict[str, Any]
    ) -> TextDocument:
        """Parse a dictionary and return a TextDocument with entities and relations

        Parameters
        ----------
        doc_line:
            Dictionary with doccano annotation

        Returns
        -------
        TextDocument
            The document with annotations
        """
        try:
            doccano_doc = _DoccanoDocRelationExtraction.from_dict(
                doc_line, client_config=self.client_config
            )
        except Exception as err:
            raise Exception(
                "Impossible to convert the document. Please check the task"
                " or the client configuration of the converter"
            ) from err

        ents_by_doccano_id = dict()
        relations = []
        for doccano_entity in doccano_doc.entities:
            text = doccano_doc.text[
                doccano_entity.start_offset : doccano_entity.end_offset
            ]
            entity = Entity(
                text=text,
                label=doccano_entity.label,
                spans=[Span(doccano_entity.start_offset, doccano_entity.end_offset)],
                metadata=dict(doccano_id=doccano_entity.id),
            )
            ents_by_doccano_id[doccano_entity.id] = entity

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    entity, self.description, source_data_items=[]
                )

        for doccano_relation in doccano_doc.relations:
            relation = Relation(
                label=doccano_relation.type,
                source_id=ents_by_doccano_id[doccano_relation.from_id].uid,
                target_id=ents_by_doccano_id[doccano_relation.to_id].uid,
                metadata=dict(doccano_id=doccano_relation.id),
            )
            relations.append(relation)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    relation, self.description, source_data_items=[]
                )

        anns = list(ents_by_doccano_id.values()) + relations
        doc = TextDocument(
            text=doccano_doc.text,
            anns=anns,
            metadata=doccano_doc.metadata,
        )

        return doc

    def _parse_doc_line_seq_labeling(self, doc_line: Dict[str, Any]) -> TextDocument:
        """Parse a dictionary and return a TextDocument with entities

        Parameters
        ----------
        doc_line:
            Dictionary with doccano annotation.

        Returns
        -------
        TextDocument
            The document with annotations
        """
        try:
            doccano_doc = _DoccanoDocSeqLabeling.from_dict(
                doc_line, client_config=self.client_config
            )
        except Exception as err:
            raise Exception(
                "Impossible to convert the document. Please check the task"
                " or the client configuration of the converter"
            ) from err

        entities = []
        for doccano_entity in doccano_doc.entities:
            text = doccano_doc.text[
                doccano_entity.start_offset : doccano_entity.end_offset
            ]
            entity = Entity(
                text=text,
                label=doccano_entity.label,
                spans=[Span(doccano_entity.start_offset, doccano_entity.end_offset)],
            )
            entities.append(entity)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    entity, self.description, source_data_items=[]
                )

        doc = TextDocument(
            text=doccano_doc.text,
            anns=entities,
            metadata=doccano_doc.metadata,
        )
        return doc

    def _parse_doc_line_text_classification(
        self, doc_line: Dict[str, Any]
    ) -> TextDocument:
        """Parse a dictionary and return a TextDocument with an attribute.

        Parameters
        ----------
        doc_line:
            Dictionary with doccano annotation.

        Returns
        -------
        TextDocument
            The document with its category
        """
        try:
            doccano_doc = _DoccanoDocTextClassification.from_dict(
                doc_line, client_config=self.client_config
            )
        except Exception as err:
            raise Exception(
                "Impossible to convert the document. Please check the task"
                " or the client configuration of the converter"
            ) from err

        attr = Attribute(label=self.attr_label, value=doccano_doc.label)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(attr, self.description, source_data_items=[])

        doc = TextDocument(text=doccano_doc.text, metadata=doccano_doc.metadata)
        doc.attrs.add(attr)
        return doc


class DoccanoOutputConverter:
    """Convert medkit files to doccano files (.JSONL) for a given task.

    For each :class:`~medkit.core.text.TextDocument` a jsonline will be created.
    """

    def __init__(
        self,
        task: DoccanoTask,
        anns_labels: Optional[List[str]] = None,
        attr_label: Optional[str] = None,
        ignore_segments: bool = True,
        include_metadata: Optional[bool] = True,
        uid: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        task:
            The doccano task for the input converter
        anns_labels:
            Labels of medkit annotations to convert into doccano annotations.
            If `None` (default) all the entities or relations will be converted.
            Useful for :class:`~.io.DoccanoTask.SEQUENCE_LABELING` or
            :class:`~.io.DoccanoTask.RELATION_EXTRACTION` converters.
        attr_label:
            The label of the medkit attribute that represents the text category.
            Useful for :class:`~.io.DoccanoTask.TEXT_CLASSIFICATION` converters.
        ignore_segments:
            If `True` medkit segments will be ignored. Only entities will be
            converted to Doccano entities.  If `False` the medkit segments will
            be converted to Doccano entities as well.
            Useful for :class:`~.io.DoccanoTask.SEQUENCE_LABELING` or
            :class:`~.io.DoccanoTask.RELATION_EXTRACTION` converters.
        include_metadata:
            Whether include medkit metadata in the converted documents
        uid:
            Identifier of the converter.
        """
        if uid is None:
            uid = generate_id()

        self.uid = uid
        self.task = task
        self.anns_labels = anns_labels
        self.attr_label = attr_label
        self.ignore_segments = ignore_segments
        self.include_metadata = include_metadata

    @property
    def description(self) -> OperationDescription:
        return OperationDescription(
            uid=self.uid,
            name=self.__class__.__name__,
            class_name=self.__class__.__name__,
            config=dict(task=self.task.value),
        )

    def save(self, docs: List[TextDocument], output_file: Union[str, Path]):
        """Convert and save a list of TextDocuments into a doccano file (.JSONL)

        Parameters
        ----------
        docs:
            List of medkit doc objects to convert
        output_file:
            Path or string of the JSONL file where to save the converted documents
        """

        output_file = Path(output_file)

        with open(output_file, mode="w", encoding="utf-8") as fp:
            for medkit_doc in docs:
                doc_line = self._convert_doc_by_task(medkit_doc)
                fp.write(json.dumps(doc_line, ensure_ascii=False) + "\n")

    def _convert_doc_by_task(self, medkit_doc: TextDocument) -> Dict[str, Any]:
        """Convert a TextDocument into a dictionary depending on the task

        Parameters
        ----------
        medkit_doc:
            Document to convert

        Returns
        -------
        Dict[str,Any]
            Dictionary with doccano annotation
        """
        if self.task == DoccanoTask.RELATION_EXTRACTION:
            return self._convert_doc_relation_extraction(medkit_doc=medkit_doc)
        if self.task == DoccanoTask.TEXT_CLASSIFICATION:
            return self._convert_doc_text_classification(medkit_doc=medkit_doc)
        if self.task == DoccanoTask.SEQUENCE_LABELING:
            return self._convert_doc_seq_labeling(medkit_doc=medkit_doc)

    def _convert_doc_relation_extraction(
        self, medkit_doc: TextDocument
    ) -> Dict[str, Any]:
        """Convert a TextDocument to a doc_line compatible
        with the doccano relation extraction task

        Parameters
        ----------
        medkit_doc:
            Document to convert, it may contain entities and relations

        Returns
        -------
        Dict[str,Any]
            Dictionary with doccano annotation. It may contain
            text, entities and relations
        """
        doccano_ents_by_medkit_uid = dict()
        doccano_relations = []

        anns_by_type = get_anns_by_type(medkit_doc, self.anns_labels)
        medkit_segments = anns_by_type["entities"]
        if not self.ignore_segments:
            medkit_segments += anns_by_type["segments"]

        for medkit_segment in medkit_segments:
            spans = span_utils.normalize_spans(medkit_segment.spans)
            ann_id = generate_deterministic_id(medkit_segment.uid)
            entity = _DoccanoEntity(
                id=ann_id.int,
                start_offset=spans[0].start,
                end_offset=spans[-1].end,
                label=medkit_segment.label,
            )
            doccano_ents_by_medkit_uid[medkit_segment.uid] = entity

        for medkit_relation in anns_by_type["relations"]:
            subj = doccano_ents_by_medkit_uid.get(medkit_relation.source_id)
            obj = doccano_ents_by_medkit_uid.get(medkit_relation.target_id)

            if subj is None or obj is None:
                logger.warning(
                    f"Ignore relation {medkit_relation.uid}. Entity source/target was"
                    " no found"
                )
                continue

            ann_id = generate_deterministic_id(medkit_relation.uid)
            relation = _DoccanoRelation(
                id=ann_id.int,
                from_id=subj.id,
                to_id=obj.id,
                type=medkit_relation.label,
            )
            doccano_relations.append(relation)

        metadata = medkit_doc.metadata if self.include_metadata else {}

        doccano_doc = _DoccanoDocRelationExtraction(
            text=medkit_doc.text,
            entities=list(doccano_ents_by_medkit_uid.values()),
            relations=doccano_relations,
            metadata=metadata,
        )

        return doccano_doc.to_dict()

    def _convert_doc_seq_labeling(self, medkit_doc: TextDocument) -> Dict[str, Any]:
        """Convert a TextDocument to a doc_line compatible
        with the doccano sequence labeling task

        Parameters
        ----------
        medkit_doc:
            Document to convert, it may contain entities

        Returns
        -------
        Dict[str,Any]
            Dictionary with doccano annotation. It may contain
            text ans its label (a list of tuples representing entities)
        """
        anns_by_type = get_anns_by_type(medkit_doc, self.anns_labels)
        medkit_segments = anns_by_type["entities"]
        if not self.ignore_segments:
            medkit_segments += anns_by_type["segments"]
        doccano_entities = []
        for medkit_segment in medkit_segments:
            spans = span_utils.normalize_spans(medkit_segment.spans)
            entity = _DoccanoEntityTuple(
                start_offset=spans[0].start,
                end_offset=spans[-1].end,
                label=medkit_segment.label,
            )
            doccano_entities.append(entity)

        metadata = medkit_doc.metadata if self.include_metadata else {}
        doccano_doc = _DoccanoDocSeqLabeling(
            text=medkit_doc.text,
            entities=doccano_entities,
            metadata=metadata,
        )

        return doccano_doc.to_dict()

    def _convert_doc_text_classification(
        self, medkit_doc: TextDocument
    ) -> Dict[str, Any]:
        """Convert a TextDocument to a doc_line compatible with
        the doccano text classification task.

        Parameters
        ----------
        medkit_doc:
            Document to convert, it may contain at least one attribute to convert.

        Returns
        -------
        Dict[str,Any]
            Dictionary with doccano annotation. It may contain
            text ans its label (a category(str))
        """
        attributes = medkit_doc.attrs.get(label=self.attr_label)

        if not attributes:
            raise KeyError(
                "The attribute with the corresponding text class was not found. Check"
                f" the 'attr_label' for this converter, {self.attr_label} was provided."
            )

        metadata = medkit_doc.metadata if self.include_metadata else {}
        doccano_doc = _DoccanoDocTextClassification(
            text=medkit_doc.text,
            label=attributes[0].value,
            metadata=metadata,
        )
        return doccano_doc.to_dict()
