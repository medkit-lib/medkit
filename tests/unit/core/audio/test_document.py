from pathlib import Path

import pytest

from medkit.core import generate_id
from medkit.core.audio import FileAudioBuffer, MemoryAudioBuffer, Segment, Span
from medkit.core.audio.document import AudioDocument
from tests.audio_utils import generate_silence

_SAMPLE_RATE = 4000

# TODO remove tests redundant with test_annotation_container


def test_basic():
    """Basic AudioDocument behavior"""
    audio = MemoryAudioBuffer(signal=generate_silence(0.5, _SAMPLE_RATE), sample_rate=_SAMPLE_RATE)
    doc = AudioDocument(audio=audio)
    assert doc.audio == audio

    duration = audio.duration
    # add speech segment on 1st quarter
    span_1 = Span(0.0, duration / 2)
    speech_audio_1 = audio.trim_duration(end_time=span_1.end)
    speech_seg_1 = Segment(
        label="speech",
        span=span_1,
        audio=speech_audio_1,
    )
    doc.anns.add(speech_seg_1)

    # add noise segment on 2d quarter
    span_2 = Span(duration / 4, duration / 2)
    noise_audio = audio.trim_duration(start_time=span_2.start, end_time=span_2.end)
    noise_seg = Segment(
        label="noise",
        span=span_2,
        audio=noise_audio,
    )
    doc.anns.add(noise_seg)

    # add speech segment on 2d half
    span_3 = Span(duration / 2, duration)
    speech_audio_2 = audio.trim_duration(start_time=span_3.start)
    speech_seg_2 = Segment(
        label="speech",
        span=span_3,
        audio=speech_audio_2,
    )
    doc.anns.add(speech_seg_2)

    assert doc.anns.get(label="speech") == [speech_seg_1, speech_seg_2]
    assert doc.anns.get(label="noise") == [noise_seg]

    assert doc.anns.get_by_id(speech_seg_1.uid) == speech_seg_1
    assert doc.anns.get_by_id(speech_seg_2.uid) == speech_seg_2
    assert doc.anns.get_by_id(noise_seg.uid) == noise_seg


def test_raw_segment():
    """Handling of raw audio segment"""
    audio = MemoryAudioBuffer(signal=generate_silence(0.5, _SAMPLE_RATE), sample_rate=_SAMPLE_RATE)

    # raw audio segment automatically created
    doc_with_raw_audio = AudioDocument(audio=audio)
    raw_seg = doc_with_raw_audio.raw_segment
    assert raw_seg.audio == audio

    # also available through get() and get_by_id()
    assert doc_with_raw_audio.anns.get(label=AudioDocument.RAW_LABEL) == [raw_seg]
    assert doc_with_raw_audio.anns.get_by_id(raw_seg.uid) == raw_seg
    # but not included in full annotations list
    assert raw_seg not in doc_with_raw_audio.anns

    # docs with same ids should have raw audio segments with same uid
    uid = generate_id()
    doc_1 = AudioDocument(uid=uid, audio=audio)
    doc_2 = AudioDocument(uid=uid, audio=audio)
    assert doc_1.raw_segment.uid == doc_2.raw_segment.uid

    # manually adding raw audio segment is forbidden
    doc = AudioDocument(audio=audio)
    seg = Segment(label=AudioDocument.RAW_LABEL, audio=audio, span=Span(0.0, audio.duration))
    with pytest.raises(RuntimeError, match=r"Cannot add annotation with reserved label .*"):
        doc.anns.add(seg)


def test_from_dir():
    dir_ = Path("tests/data/audio")
    docs = AudioDocument.from_dir(dir_, pattern="dialog*.ogg")
    assert len(docs) == 2
    assert docs[0].metadata["path_to_audio"] == str((dir_ / "dialog.ogg").absolute())
    assert docs[0].audio == FileAudioBuffer(path=dir_ / "dialog.ogg")
    assert docs[1].audio == FileAudioBuffer(path=dir_ / "dialog_long.ogg")
