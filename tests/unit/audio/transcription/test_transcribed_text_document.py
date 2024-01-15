from medkit.audio.transcription.transcribed_text_document import TranscribedTextDocument
from medkit.core import generate_id
from medkit.core.audio import Span as AudioSpan
from medkit.core.text import ModifiedSpan as ModifiedTextSpan
from medkit.core.text import Span as TextSpan

_AUDIO_SPAN_1 = AudioSpan(1.0, 7.0)
_AUDIO_SPAN_2 = AudioSpan(12.0, 19.0)


def _get_transcribed_doc():
    text = "BOB: Hello Alice.\nALICE: Hello Bob, how are you?"
    text_spans_to_audio_spans = {
        TextSpan(5, 17): _AUDIO_SPAN_1,
        TextSpan(25, 48): _AUDIO_SPAN_2,
    }
    return TranscribedTextDocument(
        text=text,
        audio_doc_id=generate_id(),
        text_spans_to_audio_spans=text_spans_to_audio_spans,
    )


def test_get_containing_audio_spans():
    doc = _get_transcribed_doc()

    # single text span fully contained in transcription text spans
    assert doc.get_containing_audio_spans([TextSpan(11, 16)]) == [_AUDIO_SPAN_1]
    assert doc.get_containing_audio_spans([TextSpan(32, 35)]) == [_AUDIO_SPAN_2]
    # multiple text spans, each contained in different transcription text span
    assert doc.get_containing_audio_spans([TextSpan(5, 16), TextSpan(26, 35)]) == [
        _AUDIO_SPAN_1,
        _AUDIO_SPAN_2,
    ]
    # single text span overlapping with multiple transcription text spans
    assert doc.get_containing_audio_spans([TextSpan(5, 35)]) == [
        _AUDIO_SPAN_1,
        _AUDIO_SPAN_2,
    ]
    # modified span
    modified_span = ModifiedTextSpan(length=5, replaced_spans=[TextSpan(32, 35)])
    assert doc.get_containing_audio_spans([modified_span]) == [_AUDIO_SPAN_2]
