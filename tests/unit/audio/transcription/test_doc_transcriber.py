from medkit.audio.transcription.doc_transcriber import DocTranscriber
from medkit.core import Attribute, Operation, ProvTracer
from medkit.core.audio import AudioDocument, MemoryAudioBuffer
from medkit.core.audio import Segment as AudioSegment
from medkit.core.audio import Span as AudioSpan
from medkit.core.text import Span as TextSpan
from tests.audio_utils import generate_silence

_SAMPLE_RATE = 4000
_AUDIO_LABEL = "speech"
_TEXT_LABEL = "section"

_FULL_AUDIO = MemoryAudioBuffer(
    signal=generate_silence(5.0, _SAMPLE_RATE),
    sample_rate=_SAMPLE_RATE,
)


class _MockTranscriber(Operation):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.output_label = "transcribed_text"

    def run(self, segments):
        for segment in segments:
            self.count += 1
            text = f"This is transcribed text number {self.count}."
            attr = Attribute(label=self.output_label, value=text)
            segment.attrs.add(attr)


def _get_audio_segment(audio_span):
    signal = generate_silence(audio_span.length, _SAMPLE_RATE)
    audio = MemoryAudioBuffer(signal, _SAMPLE_RATE)
    return AudioSegment(label=_AUDIO_LABEL, span=audio_span, audio=audio)


def _get_audio_doc(audio_spans):
    audio_doc = AudioDocument(audio=_FULL_AUDIO)
    for audio_span in audio_spans:
        audio_seg = _get_audio_segment(audio_span)
        audio_doc.anns.add(audio_seg)
    return audio_doc


def test_basic():
    """Basic behavior"""
    # build 2 audio docs with 2 and 1 spans
    audio_span_1 = AudioSpan(0.0, 0.5)
    audio_span_2 = AudioSpan(1.0, 2.5)
    audio_doc_1 = _get_audio_doc([audio_span_1, audio_span_2])
    audio_span_3 = AudioSpan(0.5, 2.0)
    audio_doc_2 = _get_audio_doc([audio_span_3])
    audio_docs = [audio_doc_1, audio_doc_2]

    doc_transcriber = DocTranscriber(
        input_label=_AUDIO_LABEL,
        output_label=_TEXT_LABEL,
        transcription_operation=_MockTranscriber(),
    )
    text_docs = doc_transcriber.run(audio_docs)
    assert len(text_docs) == len(audio_docs)

    # 1st text doc
    text_doc_1 = text_docs[0]
    # reconstructed full text is as expected
    expected_text = "This is transcribed text number 1.\nThis is transcribed text number 2."
    assert text_doc_1.text == expected_text
    # reference to original audio doc
    assert text_doc_1.audio_doc_id == audio_doc_1.uid
    # mapping to audio spans in original doc
    text_span_1 = TextSpan(0, 34)
    text_span_2 = TextSpan(35, 69)
    assert text_doc_1.text_spans_to_audio_spans == {
        text_span_1: audio_span_1,
        text_span_2: audio_span_2,
    }

    # has all expected segments
    text_segs_1 = text_doc_1.anns.get(label=_TEXT_LABEL)
    assert len(text_segs_1) == len(audio_doc_1.anns.get(label=_AUDIO_LABEL))

    # segments have corresponding texts and spans
    text_seg_1 = text_segs_1[0]
    assert text_seg_1.label == _TEXT_LABEL
    assert text_seg_1.text == "This is transcribed text number 1."
    assert text_seg_1.spans == [text_span_1]

    text_seg_2 = text_segs_1[1]
    assert text_seg_2.label == _TEXT_LABEL
    assert text_seg_2.text == "This is transcribed text number 2."
    assert text_seg_2.spans == [text_span_2]

    # 2d text doc
    text_doc_2 = text_docs[1]
    # reconstructed full text is as expected
    assert text_doc_2.text == "This is transcribed text number 3."
    # reference to original audio doc
    assert text_doc_2.audio_doc_id == audio_doc_2.uid
    # mapping to audio spans in original doc
    text_span_3 = TextSpan(0, 34)
    assert text_doc_2.text_spans_to_audio_spans == {
        text_span_3: audio_span_3,
    }

    # has all expected segments
    text_segs_2 = text_doc_2.anns.get(label=_TEXT_LABEL)
    assert len(text_segs_2) == len(audio_doc_2.anns.get(label=_AUDIO_LABEL))


def test_prov():
    """Generated provenance nodes"""
    audio_span_1 = AudioSpan(0.0, 0.5)
    audio_span_2 = AudioSpan(1.0, 2.5)
    audio_doc = _get_audio_doc([audio_span_1, audio_span_2])

    doc_transcriber = DocTranscriber(
        input_label=_AUDIO_LABEL,
        output_label=_TEXT_LABEL,
        transcription_operation=_MockTranscriber(),
    )
    prov_tracer = ProvTracer()
    doc_transcriber.set_prov_tracer(prov_tracer)
    text_doc = doc_transcriber.run([audio_doc])[0]
    text_segs = text_doc.anns.get(label=_TEXT_LABEL)
    assert len(text_segs) == 2

    # data item uid and operation uid are correct
    text_seg_1 = text_segs[0]
    prov_1 = prov_tracer.get_prov(text_seg_1.uid)
    assert prov_1.data_item == text_seg_1
    assert prov_1.op_desc == doc_transcriber.description

    # each text segment has corresponding transcription attribute of voice segment as source
    audios_segs = audio_doc.anns.get(label=_AUDIO_LABEL)
    audio_seg_1 = audios_segs[0]
    transcription_attr_1 = audio_seg_1.attrs.get(label="transcribed_text")[0]
    assert prov_1.source_data_items == [transcription_attr_1]

    text_seg_2 = text_segs[1]
    prov_2 = prov_tracer.get_prov(text_seg_2.uid)
    audio_seg_2 = audios_segs[1]
    transcription_attr_2 = audio_seg_2.attrs.get(label="transcribed_text")[0]
    assert prov_2.source_data_items == [transcription_attr_2]


def test_attrs_to_copy():
    """Copying of audio segments attributes to text segments"""
    audio_span = AudioSpan(0.0, 0.5)

    audio_seg = _get_audio_segment(audio_span)
    # copied attribute
    speaker_attr = Attribute(label="speaker", value="Bob")
    audio_seg.attrs.add(speaker_attr)
    # uncopied attribute
    audio_seg.attrs.add(Attribute(label="loud", value=True))

    audio_doc = AudioDocument(audio=_FULL_AUDIO)
    audio_doc.anns.add(audio_seg)

    doc_transcriber = DocTranscriber(
        input_label=_AUDIO_LABEL,
        output_label=_TEXT_LABEL,
        transcription_operation=_MockTranscriber(),
        attrs_to_copy=["speaker"],
    )
    text_doc = doc_transcriber.run([audio_doc])[0]
    text_seg = text_doc.anns.get(label=_TEXT_LABEL)[0]
    # only negation attribute was copied
    speaker_attrs = text_seg.attrs.get(label="speaker")
    assert len(speaker_attrs) == 1
    assert len(text_seg.attrs.get(label="loud")) == 0

    # copied attribute has same value but new id
    copied_speaker_attr = speaker_attrs[0]
    assert copied_speaker_attr.value == speaker_attr.value
    assert copied_speaker_attr.uid != speaker_attr.uid


class _CustomDocTranscriber(DocTranscriber):
    def augment_full_text_for_next_segment(self, full_text, segment_text, audio_segment):
        # retrieve speaker name from audio segment attrs and include it in full text
        if len(full_text) > 0:
            full_text += "\n\n"
        speaker = audio_segment.attrs.get(label="speaker")[0].value
        full_text += f"- {speaker.upper()}:\n"
        return full_text


def test_custom_full_text():
    """Overriding of full text reconstruction method"""
    # audio doc with speaker name attributes on segments
    audio_doc = AudioDocument(audio=_FULL_AUDIO)
    audio_span_1 = AudioSpan(0.0, 0.5)
    audio_seg_1 = _get_audio_segment(audio_span_1)
    audio_seg_1.attrs.add(Attribute(label="speaker", value="Bob"))
    audio_doc.anns.add(audio_seg_1)
    audio_span_2 = AudioSpan(1.0, 2.5)
    audio_seg_2 = _get_audio_segment(audio_span_2)
    audio_seg_2.attrs.add(Attribute(label="speaker", value="Alice"))
    audio_doc.anns.add(audio_seg_2)

    doc_transcriber = _CustomDocTranscriber(
        input_label=_AUDIO_LABEL,
        output_label=_TEXT_LABEL,
        transcription_operation=_MockTranscriber(),
    )
    text_doc = doc_transcriber.run([audio_doc])[0]

    expected_text = "- BOB:\nThis is transcribed text number 1.\n\n- ALICE:\nThis is transcribed text number 2."
    assert text_doc.text == expected_text
