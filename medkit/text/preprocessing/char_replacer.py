from __future__ import annotations

__all__ = ["CharReplacer"]

from typing import List, Optional, Tuple

from medkit.core.operation import Operation
from medkit.core.text import Segment, span_utils
from medkit.text.preprocessing.char_rules import ALL_CHAR_RULES


class CharReplacer(Operation):
    """
    Generic character replacer to be used as pre-processing module

    This module is a non-destructive module allowing to replace selected 1-char string
    with the wanted n-chars strings.
    It respects the span modification by creating a new text-bound annotation containing
    the span modification information from input text.
    """

    def __init__(
        self,
        output_label: str,
        rules: List[Tuple[str, str]] = None,
        name: Optional[str] = None,
        uid: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        output_label
            The output label of the created annotations
        rules
            The list of replacement rules. Default: ALL_CHAR_RULES
        name:
            Name describing the pre-processing module (defaults to the class name)
        uid
            Identifier of the pre-processing module
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        if rules is None:
            rules = ALL_CHAR_RULES
        self.rules = {key: value for (key, value) in rules}

        assert not any(
            len(key) != 1 for key in self.rules.keys()
        ), "CharReplacer can only contain rules that replace 1-char string."

    def run(self, segments: List[Segment]) -> List[Segment]:
        """
        Run the module on a list of segments provided as input
        and returns a new list of segments

        Parameters
        ----------
        segments
            List of segments to process

        Returns
        -------
        List[~medkit.core.text.Segment]:
            List of new segments
        """
        return [
            processed_segment
            for segment in segments
            for processed_segment in self._process_segment_text(segment)
        ]

    def _process_segment_text(self, segment: Segment):
        ranges = []
        replacement_texts = []

        for ind, c in enumerate(segment.text):
            nc = self.rules.get(c)
            if nc is not None:
                ranges.append((ind, ind + 1))
                replacement_texts.append(nc)

        new_text, new_spans = span_utils.replace(
            text=segment.text,
            spans=segment.spans,
            ranges=ranges,
            replacement_texts=replacement_texts,
        )

        processed_text = Segment(
            label=self.output_label, spans=new_spans, text=new_text
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(
                processed_text, self.description, source_data_items=[segment]
            )

        yield processed_text
