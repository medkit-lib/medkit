__all__ = [
    "load_audio_document",
    "load_audio_documents",
    "load_audio_anns",
    "save_audio_document",
    "save_audio_documents",
    "save_audio_anns",
]

import json
import warnings
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from medkit.core.audio import AudioDocument, Segment
from medkit.io.medkit_json._common import ContentType, build_header, check_header

_DOC_ANNS_SUFFIX = "_anns.jsonl"


def load_audio_document(
    input_file: Union[str, Path],
    anns_input_file: Optional[Union[str, Path]] = None,
    encoding: Optional[str] = "utf-8",
) -> AudioDocument:
    """
    Load an audio document from a medkit-json file generated with
    :func:`~medkit.io.medkit_json.save_audio_document`

    Parameters
    ----------
    input_file:
        Path to the medkit-json file containing the document
    anns_input_file:
        Optional medkit-json file containing separate annotations of the
        document.
    encoding:
        Optional encoding of `input_file` and `anns_input_file`

    Returns
    -------
    AudioDocument
        The audio document in the file
    """

    input_file = Path(input_file)

    with open(input_file, encoding=encoding) as fp:
        data = json.load(fp)
    check_header(data, ContentType.AUDIO_DOCUMENT)
    doc = AudioDocument.from_dict(data["content"])

    if anns_input_file is not None:
        for ann in load_audio_anns(anns_input_file, encoding=encoding):
            doc.anns.add(ann)

    return doc


def load_audio_documents(
    input_file: Union[str, Path], encoding: Optional[str] = "utf-8"
) -> Iterator[AudioDocument]:
    """
    Load audio documents from a medkit-json file generated with
    :func:`~medkit.io.medkit_json.save_audio_documents`

    Parameters
    ----------
    input_file:
        Path to the medkit-json file containing the documents
    encoding:
        Optional encoding of `input_file`

    Returns
    -------
    Iterator[AudioDocument]
        An iterator to the audio documents in the file
    """

    input_file = Path(input_file)

    with open(input_file, encoding=encoding) as fp:
        line = fp.readline()
        data = json.loads(line)
        check_header(data, ContentType.AUDIO_DOCUMENT_LIST)

        for line in fp:
            doc_data = json.loads(line)
            doc = AudioDocument.from_dict(doc_data)
            yield doc


def load_audio_anns(
    input_file: Union[str, Path], encoding: Optional[str] = "utf-8"
) -> Iterator[Segment]:
    """
    Load audio annotations from a medkit-json file generated with
    :func:`~medkit.io.medkit_json.save_audio_anns`

    Parameters
    ----------
    input_file:
        Path to the medkit-json file containing the annotations
    encoding:
        Optional encoding of `input_file`

    Returns
    -------
    Iterator[Segment]
        An iterator to the audio annotations in the file
    """

    input_file = Path(input_file)

    with open(input_file, encoding=encoding) as fp:
        line = fp.readline()
        data = json.loads(line)
        check_header(data, ContentType.AUDIO_ANNOTATION_LIST)

        for line in fp:
            ann_data = json.loads(line)
            ann = Segment.from_dict(ann_data)
            yield ann


def save_audio_document(
    doc: AudioDocument,
    output_file: Union[str, Path],
    split_anns: bool = False,
    anns_output_file: Optional[Union[str, Path]] = None,
    encoding: Optional[str] = "utf-8",
):
    """
    Save an audio document into a medkit-json file.

    Parameters
    ----------
    doc:
        The audio document to save
    output_file:
        Path of the generated medkit-json file
    split_anns:
        If True, the annotations will be saved in a separate medkit-json file
        instead of being included in the main document file
    anns_output_file:
        Path of the medkit-json file storing the annotations if `split_anns` is True.
        If not provided, `output_file` will be used with an extra "_anns" suffix.
    encoding:
        Optional encoding of `output_file` and `anns_output_file`
    """

    output_file = Path(output_file)
    anns_output_file = Path(anns_output_file) if anns_output_file is not None else None

    if not split_anns and anns_output_file is not None:
        warnings.warn(
            "anns_output_file provided but split_anns is False so it will not be used"
        )

    data = build_header(content_type=ContentType.AUDIO_DOCUMENT)
    data["content"] = doc.to_dict(with_anns=not split_anns)
    with open(output_file, mode="w", encoding=encoding) as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

    if split_anns:
        if anns_output_file is None:
            anns_output_file = output_file.with_suffix(_DOC_ANNS_SUFFIX)
        save_audio_anns(doc.anns, anns_output_file, encoding=encoding)


def save_audio_documents(
    docs: Iterable[AudioDocument],
    output_file: Union[str, Path],
    encoding: Optional[str] = "utf-8",
):
    """
    Save audio documents into a medkit-json file.

    Parameters
    ----------
    docs:
        The audio documents to save
    output_file:
        Path of the generated medkit-json file
    encoding:
        Optional encoding of `output_file`
    """

    output_file = Path(output_file)

    header = build_header(content_type=ContentType.AUDIO_DOCUMENT_LIST)
    with open(output_file, mode="w", encoding=encoding) as fp:
        fp.write(json.dumps(header, ensure_ascii=False) + "\n")

        for doc in docs:
            doc_data = doc.to_dict()
            fp.write(json.dumps(doc_data, ensure_ascii=False) + "\n")


def save_audio_anns(
    anns: Iterable[Segment],
    output_file: Union[str, Path],
    encoding: Optional[str] = "utf-8",
):
    """
    Save audio annotations into a medkit-json file.

    Parameters
    ----------
    docs:
        The audio annotations to save
    output_file:
        Path of the generated medkit-json file
    encoding:
        Optional encoding of `output_file`
    """

    output_file = Path(output_file)

    header = build_header(content_type=ContentType.AUDIO_ANNOTATION_LIST)
    with open(output_file, mode="w", encoding=encoding) as fp:
        fp.write(json.dumps(header, ensure_ascii=False) + "\n")

        for ann in anns:
            ann_data = ann.to_dict()
            fp.write(json.dumps(ann_data, ensure_ascii=False) + "\n")
