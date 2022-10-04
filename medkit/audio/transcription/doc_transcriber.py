from __future__ import annotations

__all__ = ["DocTranscriber", "AudioTranscriber", "AudioTranscriberDescription"]

import dataclasses
from typing import Any, Dict, List, Optional
from typing_extensions import Protocol

from medkit.audio.transcription.transcribed_document import TranscribedDocument
from medkit.core import Operation
from medkit.core.audio import AudioDocument, AudioBuffer, Segment as AudioSegment
from medkit.core.text import Segment as TextSegment, Span as TextSpan


class AudioTranscriber(Protocol):
    """Protocol for components in charge of the actual speech-to-text transcription
    to use with :class:`~.DocTranscriber`"""

    """Description of the transcriber"""
    description: AudioTranscriberDescription

    def run(self, audios: List[AudioBuffer]) -> List[str]:
        """Convert audio buffers into strings by performing speech-to-text.

        Parameters
        ----------
        audios:
            Audio buffers to converted

        Returns
        -------
        List[str]
            Text transcription for each buffer in `audios`
        """
        pass


@dataclasses.dataclass
class AudioTranscriberDescription:
    """Description of a specific instance of an audio transcriber (similarly to
    :class:`~medkit.core.operation_desc.OperationDescription`).

    Parameters
    ----------
    name:
        The name of the transcriber (typically the class name).
    config:
        The specific configuration of the instance.
    """

    name: str
    config: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(name=self.name, config=self.config)


class DocTranscriber(Operation):
    """Speech-to-text transcriber generating text documents from audio documents.

    For each text document, all audio segments with a specific label are
    converted into text segments and regrouped in a corresponding new text
    document. The text of each segment is concatenated to form the full raw text
    of the new document.

    Generated text documents are instances of
    :class:`~medkit.audio.transcription.transcribed_document.TranscribedDocument`
    (subclass of :class:`~medkit.core.text.document.TextDocument`) with
    additional info such as the id of the original audio document and a mapping
    between audio spans and text spans.

    Methods :func: `create_text_segment()` and :func:
    `augment_full_text_for_next_segment()` can be overridden to customize how
    the text segments are created and how they are concatenated to form the full
    text.

    If an audio document was initiated with a specific
    :class:`~medkit.core.Store` instance explicitly provided, then the
    corresponding text document will use the same instance. Otherwise, if the
    audio document uses its own private store, then the text document will also
    have its own private store.

    The actual transcription task is delegated to an :class:`~.AudioTranscriber`
    that must be provided.
    """

    def __init__(
        self,
        input_label: str,
        output_label: str,
        transcriber: AudioTranscriber,
        attrs_to_copy: Optional[List[str]] = None,
        op_id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        input_label:
            Label of audio segments that should be transcribed.
        output_label:
            Label of generated text segments.
        transcriber:
            Transcription component in charge of actually transforming each audio signal
            into text.
        attrs_to_copy:
            Labels of attributes that should be copied from the original audio segments
            to the transcribed text segments.
        proc_id:
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
        self.transcriber = transcriber
        self.attrs_to_copy = attrs_to_copy

    def run(self, audio_docs: List[AudioDocument]) -> List[TranscribedDocument]:
        """Return a transcribed text document for each document in `audio_docs`

        Parameters
        ----------
        audio_docs:
            Audio documents to transcribe

        Returns
        -------
        List[TranscribedDocument]:
            Transcribed text documents (once per document in `audio_docs`)
        """
        return [self._transcribe_doc(d) for d in audio_docs]

    def _transcribe_doc(self, audio_doc: AudioDocument) -> TranscribedDocument:
        # get all audio segments with specified label
        audio_segs = audio_doc.get_annotations_by_label(self.input_label)
        # transcribe them to text
        audios = [seg.audio for seg in audio_segs]
        texts = self.transcriber.run(audios)

        # rebuild full text and segments from transcribed texts
        full_text = ""
        text_segs = []
        text_spans_to_audio_spans = {}

        for text, audio_seg in zip(texts, audio_segs):
            # handle joining between segments
            full_text = self.augment_full_text_for_next_segment(
                full_text, text, audio_seg
            )
            # compute text span
            start = len(full_text)
            full_text += text
            end = len(full_text)
            span = TextSpan(start, end)
            # create TextSegment with proper span referencing full text
            text_seg = TextSegment(label=self.output_label, spans=[span], text=text)

            # copy attrs from audio segment
            for label in self.attrs_to_copy:
                for attr in audio_seg.get_attrs_by_label(label):
                    text_seg.add_attr(attr)

            text_segs.append(text_seg)

            # store mapping between text and audio span
            text_spans_to_audio_spans[span] = audio_seg.span

            # handle provenance (text segment generated from audio segment)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(text_seg, self.description, [audio_seg])

        # use shared store for new text doc if audio doc had shared store
        store = audio_doc.store if audio_doc.has_shared_store else None
        text_doc = TranscribedDocument(
            text=full_text,
            audio_doc_id=audio_doc.id,
            text_spans_to_audio_spans=text_spans_to_audio_spans,
            store=store,
        )
        for text_seg in text_segs:
            text_doc.add_annotation(text_seg)
        # TODO should this be handled by provenance?
        # if self._prov_tracer is not None:
        #     self._prov_tracer.add_prov(
        #         text_doc, self, source_data_items=[audio_doc]
        #     )
        return text_doc

    def augment_full_text_for_next_segment(
        self, full_text: str, segment_text: str, audio_segment: AudioSegment
    ) -> str:
        """Append intermediate joining text to full text before the next segment is
        concatenated to it. Override for custom behavior."""
        if len(full_text) > 0:
            full_text += "\n"
        return full_text