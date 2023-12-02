from pathlib import Path

import pytest

from medkit.core import Attribute
from medkit.core.audio import AudioDocument, FileAudioBuffer, Segment, Span

srt = pytest.importorskip("medkit.io.srt")

_AUDIO_FILE = Path("tests/data/audio/dialog.ogg")
_EXPECTED_SRT_FILE = Path("tests/data/srt/dialog.srt")


def _get_doc() -> AudioDocument:
    full_audio = FileAudioBuffer(_AUDIO_FILE)
    doc = AudioDocument(full_audio)

    span_1 = Span(start=0.161, end=2.485)
    seg_1 = Segment(
        label="turn",
        span=span_1,
        audio=full_audio.trim_duration(span_1.start, span_1.end),
    )
    attr_1 = Attribute(
        label="transcribed_text",
        value="Bonjour, excusez nous, on cherche la mairie d'Orléans",
    )
    seg_1.attrs.add(attr_1)
    doc.anns.add(seg_1)

    span_2 = Span(start=2.904, end=5.056)
    seg_2 = Segment(
        label="turn",
        span=span_2,
        audio=full_audio.trim_duration(span_2.start, span_2.end),
    )
    attr_2 = Attribute(
        label="transcribed_text",
        value="Ah la mairie d'Orléans ben il faut remonter la euh",
    )

    seg_2.attrs.add(attr_2)
    doc.anns.add(seg_2)

    # segment that is not a turn and shouldn't end up in the .srt
    span_3 = Span(start=0.161, end=5.056)
    seg_3 = Segment(
        label="voice",
        span=span_3,
        audio=full_audio.trim_duration(span_3.start, span_3.end),
    )
    doc.anns.add(seg_3)

    return doc


def test_basic(tmp_path: Path):
    doc = _get_doc()
    srt_dir = tmp_path / "srt"

    converter = srt.SRTOutputConverter()
    converter.save([doc], srt_dir, doc_names=["dialog"])

    srt_file = srt_dir / "dialog.srt"
    assert srt_file.exists()

    srt_lines = srt_file.read_text().split("\n")
    expected_srt_lines = _EXPECTED_SRT_FILE.read_text().split("\n")
    assert srt_lines == expected_srt_lines
