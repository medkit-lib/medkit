from __future__ import annotations

__all__ = ["SentenceTokenizer"]

import re
from typing import Iterator

from medkit.core.text import Segment, SegmentationOperation, span_utils


class SentenceTokenizer(SegmentationOperation):
    """Sentence segmentation annotator based on end punctuation rules"""

    _DEFAULT_LABEL = "sentence"
    _DEFAULT_PUNCT_CHARS = (".", ";", "?", "!")

    def __init__(
        self,
        output_label: str = _DEFAULT_LABEL,
        punct_chars: tuple[str, ...] = _DEFAULT_PUNCT_CHARS,
        keep_punct: bool = False,
        split_on_newlines: bool = True,
        attrs_to_copy: list[str] | None = None,
        uid: str | None = None,
    ):
        """Instantiate the sentence tokenizer

        Parameters
        ----------
        output_label : str, optional
            The output label of the created annotations.
        punct_chars : tuple of str, optional
            The set of characters corresponding to end punctuations.
        keep_punct: bool, optional
            If True, the end punctuations are kept in the detected sentence.
            If False, the sentence text does not include the end punctuations.
        split_on_newlines : bool, default=True
            Whether to consider that newlines characters are sentence boundaries or not.
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the input segment
            to the derived segment. For example, useful for propagating section name.
        uid : str, optional
            Identifier of the tokenizer
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.output_label = output_label
        self.punct_chars = punct_chars
        self.keep_punct = keep_punct
        self.split_on_newlines = split_on_newlines
        self.attrs_to_copy = attrs_to_copy

        # pre-compile patterns
        self._newline_pattern = re.compile(r" *(?P<content>[^\n\r]+) *(?P<separator>[\n\r]+|$)")
        punct_string = re.escape("".join(self.punct_chars))
        self._punct_pattern = re.compile(rf" *(?P<content>[^{punct_string}]+) *(?P<separator>[{punct_string}]+|$)")

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Return sentences detected in `segments`.

        Parameters
        ----------
        segments : list of Segment
            List of segments into which to look for sentences

        Returns
        -------
        list of Segment
            Sentences segments found in `segments`
        """
        return [sentence for segment in segments for sentence in self._find_sentences_in_segment(segment)]

    def _find_sentences_in_segment(self, segment: Segment) -> Iterator[Segment]:
        # split on newlines (discarding newline chars) then split each line on punct chars
        if self.split_on_newlines:
            for line_start, line_end in self._split_text(segment.text, self._newline_pattern, keep_separator=False):
                sub_text = segment.text[line_start:line_end]
                for sub_start, sub_end in self._split_text(
                    sub_text, self._punct_pattern, keep_separator=self.keep_punct
                ):
                    start = line_start + sub_start
                    end = line_start + sub_end
                    yield self._build_sentence(segment, range_=(start, end))
        # or split directly on punct chars
        else:
            for start, end in self._split_text(segment.text, self._punct_pattern, keep_separator=self.keep_punct):
                yield self._build_sentence(segment, range_=(start, end))

    @staticmethod
    def _split_text(text: str, pattern: re.Pattern, keep_separator: bool) -> Iterator[tuple[int, int]]:
        for match in pattern.finditer(text):
            start = match.start("content")
            end = match.end("separator") if keep_separator else match.end("content")
            has_letters = re.search(r"\w", text[start:end])
            if end > start and has_letters:
                yield start, end

    def _build_sentence(self, source_segment: Segment, range_: tuple[int, int]) -> Segment:
        text, spans = span_utils.extract(
            text=source_segment.text,
            spans=source_segment.spans,
            ranges=[range_],
        )

        sentence = Segment(
            label=self.output_label,
            spans=spans,
            text=text,
        )

        # Copy inherited attributes
        for label in self.attrs_to_copy:
            for attr in source_segment.attrs.get(label=label):
                copied_attr = attr.copy()
                sentence.attrs.add(copied_attr)
                # handle provenance
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(copied_attr, self.description, [attr])

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(sentence, self.description, source_data_items=[source_segment])

        return sentence
