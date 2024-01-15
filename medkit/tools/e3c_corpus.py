"""This module aims to provide facilities for accessing data from e3c corpus.

**Version** : 2.0.0
**License**: The E3C corpus is released under Creative Commons NonCommercial license
(CC BY-NC).

**Github**: https://github.com/hltfbk/E3C-Corpus

**Reference**

B. magnini, B. Altuna, A. Lavelli, M. Speranza, and R. Zanoli. 2020.
The E3C Project: Collection and Annotation of a Multilingual Corpus of Clinical Cases.
In Proceedings of the Seventh Italian Conference on Computational Linguistics, Bologna,
Italy, December.
Associazione Italiana di Linguistica Computazionale.
"""
from __future__ import annotations

__all__ = [
    "load_document",
    "load_data_collection",
    "convert_data_collection_to_medkit",
    "load_annotated_document",
    "load_data_annotation",
    "convert_data_annotation_to_medkit",
    "SENTENCE_LABEL",
    "CLINENTITY_LABEL",
]

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

from medkit.core import generate_deterministic_id
from medkit.core.text import Entity, Segment, Span, TextDocument, UMLSNormAttribute
from medkit.io.medkit_json import save_text_documents

logger = logging.getLogger(__name__)


SENTENCE_LABEL = "sentence"
"""
Label used by medkit for annotated sentences of E3C corpus
"""
CLINENTITY_LABEL = "disorder"
"""
Label used by medkit for annotated clinical entities of E3C corpus
"""


@dataclass
class E3CDocument:
    """Represents the data structure of a json document
    in data collection folder of the E3C corpus
    """

    authors: list[dict]  # list of {'author': '<name>'}
    doi: str
    publication_date: str
    id: str
    url: str
    source: str
    source_url: str
    licence: str
    language: str
    type: str
    description: str
    text: str

    def extract_metadata(self) -> dict:
        """Returns the metadata dict for medkit text document"""
        dict_repr = self.__dict__.copy()
        dict_repr.pop("text")
        return dict_repr


def load_document(filepath: str | Path, encoding: str = "utf-8") -> TextDocument:
    """Load a E3C corpus document (json document) as medkit text document.
    For example, one in data collection folder.
    Document id is always kept in medkit document metadata.

    Parameters
    ----------
    filepath : str or Path
        The path to the json file of the E3C corpus
    encoding : str, default="utf-8"
        The encoding of the file. Default: 'utf-8'

    Returns
    -------
    TextDocument
        The corresponding medkit text document
    """
    with Path(filepath).open(encoding=encoding) as f:
        doc = E3CDocument(**json.load(f))

        uid = str(generate_deterministic_id(doc.id))
        return TextDocument(text=doc.text, uid=uid, metadata=doc.extract_metadata())


def load_data_collection(dir_path: Path | str, encoding: str = "utf-8") -> Iterator[TextDocument]:
    """Load the E3C corpus data collection as medkit text documents

    Parameters
    ----------
    dir_path : str or Path
        The path to the E3C corpus data collection directory containing the json files
        (e.g., /tmp/E3C-Corpus-2.0.0/data_collection/French/layer1)
    encoding : str, default="utf-8"
        The encoding of the files. Default: 'utf-8'

    Returns
    -------
    iterator of TextDocument
        An iterator on corresponding medkit text documents
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        msg = "%s is not a directory or does not exist"
        raise FileNotFoundError(msg, dir_path)

    filepaths = sorted(dir_path.glob("*.json"))
    if not filepaths:
        logger.warning(
            "No .json document found inside '%s', make sure you are passing a layer"
            " subdirectory inside data_collection",
            dir_path,
        )
    for filepath in filepaths:
        yield load_document(filepath, encoding=encoding)


def convert_data_collection_to_medkit(dir_path: Path | str, output_file: str | Path, encoding: str | None = "utf-8"):
    """Convert E3C corpus data collection to medkit jsonl file

    Parameters
    ----------
    dir_path : str or Path
        The path to the E3C corpus data collection directory containing the json files
        (e.g., /tmp/E3C-Corpus-2.0.0/data_collection/French/layer1)
    output_file : str or Path
        The medkit jsonl output file which will contain medkit text documents
    encoding : str, default="utf-8"
        The encoding of the files. Default: 'utf-8'
    """
    docs = load_data_collection(dir_path=dir_path, encoding=encoding)
    save_text_documents(docs=docs, output_file=output_file, encoding=encoding)


def load_annotated_document(filepath: str | Path, encoding: str = "utf-8", keep_sentences=False) -> TextDocument:
    """Load a E3C corpus annotated document (xml document) as medkit text document.
    For example, one in data annotation folder.
    Each annotation id is always kept in corresponding medkit element metadata.

    For the time being, only supports 'CLINENTITY' annotations.
    'SENTENCE' annotations may be also loaded.

    Parameters
    ----------
    filepath : str | Path
        The path to the xml file of the E3C corpus
    encoding : str, default="utf-8"
        The encoding of the file. Default: 'utf-8'
    keep_sentences : bool, default=False
        Whether to load sentences into medkit documents.

    Returns
    -------
    TextDocument
        The corresponding medkit text document
    """
    xml_parser = ElementTree.XMLParser(encoding=encoding)
    root = ElementTree.parse(filepath, parser=xml_parser).getroot()
    # get xml namespaces
    ns = dict([node for _, node in ElementTree.iterparse(filepath, events=["start-ns"])])
    metadata = root.find("custom:METADATA", ns).attrib
    text = root.find("cas:Sofa", ns).attrib.get("sofaString", "")
    doc = E3CDocument(
        authors=[{"author": author.strip()} for author in metadata["docAuthor"].split(";")],
        doi=metadata["docDOI"],
        publication_date=metadata["docTime"],
        id=metadata["docName"],
        url=metadata["docUrl"],
        source=metadata["docSource"],
        source_url=metadata["docSourceUrl"],
        licence=metadata["docLicense"],
        language=metadata["docLanguage"],
        type=metadata["pubType"],
        description=metadata["note"],
        text=text,
    )

    # create medkit text document
    doc_uid = str(generate_deterministic_id(doc.id))
    medkit_doc = TextDocument(text=doc.text, uid=doc_uid, metadata=doc.extract_metadata())

    # parse sentences if wanted by user
    if keep_sentences:
        for elem in root.findall("type4:Sentence", ns):
            sentence = elem.attrib
            span = Span(int(sentence["begin"]), int(sentence["end"]))
            sentence_uid = sentence["{http://www.omg.org/XMI}id"]

            medkit_sentence = Segment(
                uid=str(generate_deterministic_id(doc_uid + sentence_uid)),
                label=SENTENCE_LABEL,
                spans=[span],
                text=doc.text[span.start : span.end],
                metadata={"id": sentence_uid},
            )

            # attach medkit sentence to medkit document
            medkit_doc.anns.add(medkit_sentence)

    # parse clinical entities
    for elem in root.findall("custom:CLINENTITY", ns):
        clin_entity = elem.attrib
        span = Span(int(clin_entity["begin"]), int(clin_entity["end"]))
        entity_uid = clin_entity["{http://www.omg.org/XMI}id"]  # retrieve xmi:id from attributes

        medkit_entity = Entity(
            uid=str(generate_deterministic_id(doc_uid + entity_uid)),
            label=CLINENTITY_LABEL,
            spans=[span],
            text=doc.text[span.start : span.end],
            metadata={"id": entity_uid},
        )
        # add normalization attribute to medkit entity
        cui = clin_entity.get("entityID")
        if cui is not None:
            metadata = {
                "id": clin_entity.get("{http://www.omg.org/XMI}id"),
                "entityIDEN": clin_entity.get("entityIDEN"),
                "discontinuous": clin_entity.get("discontinuous"),
                "xtra": clin_entity.get("xtra"),
            }
            attr_uid = str(generate_deterministic_id("norm" + doc_uid + entity_uid))
            attr = UMLSNormAttribute(cui=cui, umls_version="", metadata=metadata, uid=str(attr_uid))
            medkit_entity.attrs.add(attr)

        else:
            logger.debug("no cui for %s", medkit_entity)

        # attach medkit entity to medkit document
        medkit_doc.anns.add(medkit_entity)

    return medkit_doc


def load_data_annotation(
    dir_path: Path | str,
    encoding: str = "utf-8",
    keep_sentences: bool = False,
) -> Iterator[TextDocument]:
    """Load the E3C corpus data annotation as medkit text documents.

    Parameters
    ----------
    dir_path : str or Path
        The path to the E3C corpus data annotation directory containing the xml files
        (e.g., /tmp/E3C-Corpus-2.0.0/data_annotation/French/layer1)
    encoding : str, default="utf-8"
        The encoding of the files. Default: 'utf-8'
    keep_sentences : bool, default=False
        Whether to load sentences into medkit documents.

    Returns
    -------
    iterator of TextDocument
        An iterator on corresponding medkit text documents
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        msg = "%s is not a directory or does not exist"
        raise FileNotFoundError(msg, dir_path)

    filepaths = sorted(dir_path.glob("*.xml"))
    if not filepaths:
        logger.warning(
            "No .xml document found inside '%s', make sure your are passing a layer"
            " subdirectory inside data_annotation",
            dir_path,
        )
    for filepath in filepaths:
        yield load_annotated_document(filepath, encoding=encoding, keep_sentences=keep_sentences)


def convert_data_annotation_to_medkit(
    dir_path: Path | str,
    output_file: str | Path,
    encoding: str | None = "utf-8",
    keep_sentences: bool = False,
):
    """Convert E3C corpus data annotation to medkit jsonl file.

    Parameters
    ----------
    dir_path : str or Path
        The path to the E3C corpus data collection directory containing the json files
        (e.g., /tmp/E3C-Corpus-2.0.0/data_collection/French/layer1)
    output_file : str or Path
        The medkit jsonl output file which will contain medkit text documents
    encoding : str, default="utf-8"
        The encoding of the files. Default: 'utf-8'
    keep_sentences : bool, default=False
        Whether to load sentences into medkit documents.
    """
    docs = load_data_annotation(
        dir_path=dir_path,
        encoding=encoding,
        keep_sentences=keep_sentences,
    )
    save_text_documents(docs=docs, output_file=output_file, encoding=encoding)
