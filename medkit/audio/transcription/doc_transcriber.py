from __future__ import annotations

__all__ = ["DocTranscriber", "TranscriptionOperation"]

from typing import TYPE_CHECKING

from typing_extensions import Protocol

from medkit.audio.transcription.transcribed_text_document import TranscribedTextDocument
from medkit.core import Operation
from medkit.core.text import Segment as TextSegment
from medkit.core.text import Span as TextSpan

if TYPE_CHECKING:
    from medkit.core.audio import AudioDocument
    from medkit.core.audio import Segment as AudioSegment


class TranscriptionOperation(Protocol):
    """Protocol for operations in charge of the actual speech-to-text transcription
    to use with :class:`~.DocTranscriber`
    """

    output_label: str
    """
    Label to use for generated transcription attributes
    """

    def run(self, segments: list[AudioSegment]):
        """Add a transcription attribute to each segment with a text value
        containing the transcribed text.

        Parameters
        ----------
        segments: list of AudioSegment
            List of segments to transcribe
        """


class DocTranscriber(Operation):
    """Speech-to-text transcriber generating text documents from audio documents.

    For each text document, all audio segments with a specific label are
    converted into text segments and regrouped in a corresponding new text
    document. The text of each segment is concatenated to form the full raw text
    of the new document.

    Generated text documents are instances of
    :class:`~medkit.audio.transcription.transcribed_text_document.TranscribedTextDocument`
    (subclass of :class:`~medkit.core.text.document.TextDocument`) with
    additional info such as the identifier of the original audio document and a mapping
    between audio spans and text spans.

    Methods :func: `create_text_segment()` and :func:
    `augment_full_text_for_next_segment()` can be overridden to customize how
    the text segments are created and how they are concatenated to form the full
    text.

    The actual transcription task is delegated to a
    :class:`~.TranscriptionOperation` that must be provided, for instance
    :class`~medkit.audio.transcription.hf_transcriber.HFTranscriber` or
    :class`~medkit.audio.transcription.sb_transcriber.SBTranscriber`.
    """

    def __init__(
        self,
        input_label: str,
        output_label: str,
        transcription_operation: TranscriptionOperation,
        attrs_to_copy: list[str] | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        input_label: str
            Label of audio segments that should be transcribed.
        output_label: str
            Label of generated text segments.
        transcription_operation: TranscriptionOperation
            Transcription operation in charge of actually transcribing each
            audio segment.
        attrs_to_copy: list of str, optional
            Labels of attributes that should be copied from the original audio segments
            to the transcribed text segments.
        uid: str, optional
            Identifier of the transcriber.
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.input_label = input_label
        self.output_label = output_label
        self.transcription_operation = transcription_operation
        self.attrs_to_copy = attrs_to_copy

        # label of transcription attributes attached to audio segments
        self._attr_label = self.transcription_operation.output_label

    def run(self, audio_docs: list[AudioDocument]) -> list[TranscribedTextDocument]:
        """Return a transcribed text document for each document in `audio_docs`

        Parameters
        ----------
        audio_docs: list of AudioDocument
            Audio documents to transcribe

        Returns
        -------
        list of TranscribedTextDocument:
            Transcribed text documents (once per document in `audio_docs`)
        """
        return [self._transcribe_doc(d) for d in audio_docs]

    def _transcribe_doc(self, audio_doc: AudioDocument) -> TranscribedTextDocument:
        # get all audio segments with specified label
        audio_segs = audio_doc.anns.get(label=self.input_label)
        # transcribe them to text
        self.transcription_operation.run(audio_segs)

        # rebuild full text and segments from transcribed texts
        full_text = ""
        text_segs = []
        text_spans_to_audio_spans = {}

        for audio_seg in audio_segs:
            # retrieve transcription attribute
            transcription_attr = audio_seg.attrs.get(label=self._attr_label)[0]
            text = transcription_attr.value

            # handle joining between segments
            full_text = self.augment_full_text_for_next_segment(full_text, text, audio_seg)

            # compute text span
            start = len(full_text)
            full_text += text
            end = len(full_text)
            span = TextSpan(start, end)
            # create TextSegment with proper span referencing full text
            text_seg = TextSegment(label=self.output_label, spans=[span], text=text)

            # copy attrs from audio segment
            for label in self.attrs_to_copy:
                for attr in audio_seg.attrs.get(label=label):
                    copied_attr = attr.copy()
                    text_seg.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            text_segs.append(text_seg)

            # store mapping between text and audio span
            text_spans_to_audio_spans[span] = audio_seg.span

            # handle provenance (text segment generated from transcription attribute)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(text_seg, self.description, [transcription_attr])

        text_doc = TranscribedTextDocument(
            text=full_text,
            audio_doc_id=audio_doc.uid,
            text_spans_to_audio_spans=text_spans_to_audio_spans,
        )
        for text_seg in text_segs:
            text_doc.anns.add(text_seg)
        # TODO should this be handled by provenance?
        # if self._prov_tracer is not None:
        #     self._prov_tracer.add_prov(
        #         text_doc, self, source_data_items=[audio_doc]
        #     )
        return text_doc

    def augment_full_text_for_next_segment(self, full_text: str, segment_text: str, audio_segment: AudioSegment) -> str:
        """Append intermediate joining text to full text before the next segment is
        concatenated to it. Override for custom behavior.
        """
        if len(full_text) > 0:
            full_text += "\n"
        return full_text
