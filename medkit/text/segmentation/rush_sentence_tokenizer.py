"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[rush-sentence-tokenizer]`.
"""

from __future__ import annotations

__all__ = ["RushSentenceTokenizer"]

import re
from pathlib import Path
from typing import Iterator

from PyRuSH import RuSH

from medkit.core.text import Segment, SegmentationOperation, span_utils

_PATH_TO_DEFAULT_RULES = Path(__file__).parent / "rush_sentence_tokenizer_default_rules.tsv"


class RushSentenceTokenizer(SegmentationOperation):
    """Sentence segmentation annotator based on PyRuSH."""

    _DEFAULT_LABEL = "sentence"

    def __init__(
        self,
        output_label: str = _DEFAULT_LABEL,
        path_to_rules: str | Path | None = None,
        keep_newlines: bool = True,
        attrs_to_copy: list[str] | None = None,
        uid: str | None = None,
    ):
        """Instantiate the RuSH tokenizer

        Parameters
        ----------
        output_label: str, optional
            The output label of the created annotations.
        path_to_rules: str or Path, optional
            Path to csv or tsv file to provide to PyRuSH. If none provided,
            "rush_tokenizer_default_rules.tsv" will be used
            (corresponds to the "conf/rush_rules.tsv" in the PyRush repo)
        keep_newlines: bool, default=True
            With the default rules, newline chars are not used to split
            sentences, therefore a sentence maybe contain one or more newline chars.
            If `keep_newlines` is False, newlines will be replaced by spaces.
        attrs_to_copy: list of str, optional
            Labels of the attributes that should be copied from the input segment
            to the derived segment. For example, useful for propagating section name.
        uid: str, optional
            Identifier of the tokenizer
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        if path_to_rules is None:
            path_to_rules = _PATH_TO_DEFAULT_RULES

        self.output_label = output_label
        self.path_to_rules = path_to_rules
        self.keep_newlines = keep_newlines
        self.attrs_to_copy = attrs_to_copy
        self._rush = RuSH(str(path_to_rules))

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Return sentences detected in `segments`.

        Parameters
        ----------
        segments: list of Segment
            List of segments into which to look for sentences

        Returns
        -------
        list of Segment:
            Sentences segments found in `segments`
        """
        return [sentence for segment in segments for sentence in self._find_sentences_in_segment(segment)]

    def _find_sentences_in_segment(self, segment: Segment) -> Iterator[Segment]:
        rush_spans = self._rush.segToSentenceSpans(segment.text)
        for rush_span in rush_spans:
            text, spans = span_utils.extract(
                text=segment.text,
                spans=segment.spans,
                ranges=[(rush_span.begin, rush_span.end)],
            )

            if not self.keep_newlines:
                ranges = [m.span() for m in re.finditer(r"\n", text)]
                replacements = [" " for _ in ranges]
                text, spans = span_utils.replace(text, spans, ranges, replacements)

            sentence = Segment(
                label=self.output_label,
                spans=spans,
                text=text,
            )

            # Copy inherited attributes
            for label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=label):
                    copied_attr = attr.copy()
                    sentence.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(sentence, self.description, source_data_items=[segment])

            yield sentence
