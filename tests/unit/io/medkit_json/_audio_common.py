__all__ = [
    "build_doc",
    "build_docs",
    "build_anns",
    "DOC_JSON_FILE",
    "DOCS_JSONL_FILE",
    "ANNS_JSONL_FILE",
    "SPLIT_DOC_JSON_FILE",
    "SPLIT_DOC_ANNS_JSONL_FILE",
]

from pathlib import Path

from medkit.core import Attribute
from medkit.core.audio import AudioDocument, FileAudioBuffer, Segment, Span

DOC_JSON_FILE = Path("tests/data/medkit_json/audio_doc.json")
DOCS_JSONL_FILE = Path("tests/data/medkit_json/audio_docs.jsonl")
ANNS_JSONL_FILE = Path("tests/data/medkit_json/audio_anns.jsonl")
SPLIT_DOC_JSON_FILE = Path("tests/data/medkit_json/split_audio_doc.json")
SPLIT_DOC_ANNS_JSONL_FILE = Path("tests/data/medkit_json/split_audio_doc_anns.jsonl")

_AUDIO_FILE_1 = Path("tests/data/audio/dialog_long.ogg")
_AUDIO_FILE_2 = Path("tests/data/audio/dialog.ogg")


def build_doc():
    """Build an audio doc with 2 segments and 1 attribute on the 1st segment"""
    audio = FileAudioBuffer(_AUDIO_FILE_1)
    doc = AudioDocument(uid="d1", audio=audio)
    seg_1 = Segment(
        uid="s1",
        label="speaker_1",
        span=Span(0.7, 3.6),
        audio=audio.trim_duration(0.7, 3.6),
    )
    seg_1.attrs.add(Attribute(uid="a1", label="is_negated", value=False))
    doc.anns.add(seg_1)
    seg_2 = Segment(
        uid="s2",
        label="speaker_2",
        span=Span(4.0, 8.0),
        audio=audio.trim_duration(4.0, 8.0),
    )
    doc.anns.add(seg_2)

    return doc


def build_docs():
    """Build 2 audio docs with 1 segment each"""
    audio_1 = FileAudioBuffer(_AUDIO_FILE_1)
    doc_1 = AudioDocument(uid="d1", audio=audio_1)
    seg_1 = Segment(
        uid="s1",
        label="voice",
        span=Span(0.7, 3.6),
        audio=audio_1.trim_duration(0.7, 3.6),
    )
    doc_1.anns.add(seg_1)

    audio_2 = FileAudioBuffer(_AUDIO_FILE_2)
    doc_2 = AudioDocument(uid="d2", audio=audio_2)
    seg_2 = Segment(
        uid="s2",
        label="voice",
        span=Span(0.2, 2.6),
        audio=audio_2.trim_duration(0.2, 2.6),
    )
    doc_2.anns.add(seg_2)

    return [doc_1, doc_2]


def build_anns():
    """Build 2 segments with 1 attribute on the 1st segment"""
    audio = FileAudioBuffer(_AUDIO_FILE_1)
    seg_1 = Segment(
        uid="s1",
        label="speaker_1",
        span=Span(0.7, 3.6),
        audio=audio.trim_duration(0.7, 3.6),
    )
    seg_1.attrs.add(Attribute(uid="a1", label="is_negated", value=False))
    seg_2 = Segment(
        uid="s2",
        label="speaker_2",
        span=Span(4.0, 8.0),
        audio=audio.trim_duration(4.0, 8.0),
    )
    return [seg_1, seg_2]
