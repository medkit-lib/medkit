from pathlib import Path

from medkit.core import ProvTracer
from medkit.core.audio import FileAudioBuffer
from medkit.io import RTTMInputConverter

_RRTM_DIR = Path("tests/data/rttm")
_AUDIO_DIR = Path("tests/data/audio")


def test_basic():
    converter = RTTMInputConverter()
    docs = converter.load(_RRTM_DIR, _AUDIO_DIR, audio_ext=".ogg")
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
    speaker_attrs_1 = seg_1.attrs.get(label="speaker")
    assert len(speaker_attrs_1) == 1
    speaker_attr_1 = speaker_attrs_1[0]
    assert speaker_attr_1.label == "speaker"
    assert speaker_attr_1.value == "Alice"

    # 2d turn, different speaker
    seg_2 = segments[1]
    assert seg_2.span.start == 2.904
    assert seg_2.span.end == 5.056
    speaker_attrs_2 = seg_2.attrs.get(label="speaker")
    assert len(speaker_attrs_2) == 1
    speaker_attr_2 = speaker_attrs_2[0]
    assert speaker_attr_2.value == "Bob"


def test_prov():
    prov_tracer = ProvTracer()
    converter = RTTMInputConverter()
    converter.set_prov_tracer(prov_tracer)

    docs = converter.load(_RRTM_DIR, _AUDIO_DIR, audio_ext=".ogg")
    doc = docs[0]
    segments = doc.anns.get(label="turn")

    seg = segments[0]
    seg_prov = prov_tracer.get_prov(seg.uid)
    assert seg_prov.data_item == seg
    assert len(seg_prov.source_data_items) == 0
    assert len(seg_prov.derived_data_items) == 0
    assert seg_prov.op_desc == converter.description

    speaker_attr = seg.attrs.get(label="speaker")[0]
    attr_prov = prov_tracer.get_prov(speaker_attr.uid)
    assert attr_prov.data_item == speaker_attr
    assert len(attr_prov.source_data_items) == 0
    assert len(attr_prov.derived_data_items) == 0
    assert attr_prov.op_desc == converter.description
