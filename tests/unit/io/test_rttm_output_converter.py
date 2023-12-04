from pathlib import Path

from medkit.core import Attribute
from medkit.core.audio import AudioDocument, FileAudioBuffer, Segment, Span
from medkit.io import RTTMOutputConverter

_AUDIO_FILE = Path("tests/data/audio/dialog.ogg")
_EXPECTED_RTTM_FILE = Path("tests/data/rttm/dialog.rttm")


def _get_doc():
    full_audio = FileAudioBuffer(_AUDIO_FILE)
    doc = AudioDocument(full_audio)

    span_1 = Span(start=0.161, end=2.485)
    seg_1 = Segment(
        label="turn",
        span=span_1,
        audio=full_audio.trim_duration(span_1.start, span_1.end),
    )
    seg_1.attrs.add(Attribute(label="speaker", value="Alice"))
    doc.anns.add(seg_1)

    span_2 = Span(start=2.904, end=5.056)
    seg_2 = Segment(
        label="turn",
        span=span_2,
        audio=full_audio.trim_duration(span_2.start, span_2.end),
    )
    seg_2.attrs.add(Attribute(label="speaker", value="Bob"))
    doc.anns.add(seg_2)

    # segment that is not a turn and shouldn't end up in the .rttm
    span_3 = Span(start=0.161, end=5.056)
    seg_3 = Segment(
        label="voice",
        span=span_3,
        audio=full_audio.trim_duration(span_3.start, span_3.end),
    )
    doc.anns.add(seg_3)

    return doc


def test_basic(tmp_path):
    doc = _get_doc()
    rttm_dir = tmp_path / "rttm"

    converter = RTTMOutputConverter()
    converter.save([doc], rttm_dir, doc_names=["dialog"])

    rttm_file = rttm_dir / "dialog.rttm"
    assert rttm_file.exists()

    rttm_lines = rttm_file.read_text().split("\n")
    expected_rttm_lines = _EXPECTED_RTTM_FILE.read_text().split("\n")
    assert rttm_lines == expected_rttm_lines
