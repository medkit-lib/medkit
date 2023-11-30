from pathlib import Path

import pytest

from medkit.core import ProvTracer
from medkit.core.audio import FileAudioBuffer

srt = pytest.importorskip("medkit.io.srt")

_SRT_DIR = Path("tests/data/srt")
_AUDIO_DIR = Path("tests/data/audio")


def test_basic():
    converter = srt.SRTInputConverter()
    docs = converter.load(_SRT_DIR, _AUDIO_DIR, audio_ext=".ogg")
    assert len(docs) == 1
    doc = docs[0]
    segments = doc.anns.get(label="turn")
    assert len(segments) == 2

    # 1st turn
    seg_1 = segments[0]
    assert seg_1.label == "turn"
    assert seg_1.span.start == 0.161
    assert seg_1.span.end == 2.485
    assert isinstance(seg_1.audio, FileAudioBuffer)
    assert seg_1.audio.path == _AUDIO_DIR / "dialog.ogg"
    attrs_1 = seg_1.attrs.get(label="transcribed_text")
    assert len(attrs_1) == 1
    assert attrs_1[0].value == "Bonjour, excusez nous, on cherche la mairie d'Orléans"

    # 2d turn
    seg_2 = segments[1]
    assert seg_2.span.start == 2.904
    assert seg_2.span.end == 5.056
    attrs_2 = seg_2.attrs.get(label="transcribed_text")
    assert len(attrs_2) == 1
    assert attrs_2[0].value == "Ah la mairie d'Orléans ben il faut remonter la euh"


def test_prov():
    prov_tracer = ProvTracer()
    converter = srt.SRTInputConverter()
    converter.set_prov_tracer(prov_tracer)

    docs = converter.load(_SRT_DIR, _AUDIO_DIR, audio_ext=".ogg")
    doc = docs[0]
    segments = doc.anns.get(label="turn")

    seg = segments[0]
    seg_prov = prov_tracer.get_prov(seg.uid)
    assert seg_prov.data_item == seg
    assert len(seg_prov.source_data_items) == 0
    assert len(seg_prov.derived_data_items) == 0
    assert seg_prov.op_desc == converter.description

    attr = seg.attrs.get(label="transcribed_text")[0]
    attr_prov = prov_tracer.get_prov(attr.uid)
    assert attr_prov.data_item == attr
    assert len(attr_prov.source_data_items) == 0
    assert len(attr_prov.derived_data_items) == 0
    assert attr_prov.op_desc == converter.description
