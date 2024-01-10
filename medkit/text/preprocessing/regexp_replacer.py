from __future__ import annotations

__all__ = ["RegexpReplacer"]

import re
from typing import NamedTuple

from medkit.core.operation import Operation
from medkit.core.text import Segment, span_utils


class _Rule(NamedTuple):
    pattern_to_replace: str
    new_text: str


class RegexpReplacer(Operation):
    """Generic pattern replacer to be used as pre-processing module

    This module is a non-destructive module allowing to replace a regex pattern
    by a new text.
    It respects the span modification by creating a new text-bound annotation containing
    the span modification information from input text.
    """

    def __init__(
        self,
        output_label: str,
        rules: list[tuple[str, str]] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        output_label : str
            The output label of the created annotations
        rules : list of tuple, optional
            The list of replacement rules [(pattern_to_replace, new_text)]
        name : str, optional
            Name describing the pre-processing module (defaults to the class name)
        uid : str, optional
            Identifier of the pre-processing module
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        if rules is None:
            rules = []
        self.rules = [_Rule(*rule) for rule in rules]

        regex_rules = ["(" + rule.pattern_to_replace + ")" for rule in self.rules]
        regex_rule = r"|".join(regex_rules)

        self._pattern = re.compile(regex_rule)

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Run the module on a list of segments provided as input
        and returns a new list of segments

        Parameters
        ----------
        segments : list of Segment
            List of segments to normalize

        Returns
        -------
        list of Segment
            List of normalized segments
        """
        return [norm_segment for segment in segments for norm_segment in self._normalize_segment_text(segment)]

    def _normalize_segment_text(self, segment: Segment):
        ranges = []
        replacement_texts = []

        for match in self._pattern.finditer(segment.text):
            ranges.append(match.span())
            for index in range(len(self.rules)):
                if match.groups()[index] is not None:
                    replacement_texts.append(self.rules[index].new_text)
                    break

        new_text, new_spans = span_utils.replace(
            text=segment.text,
            spans=segment.spans,
            ranges=ranges,
            replacement_texts=replacement_texts,
        )

        normalized_text = Segment(label=self.output_label, spans=new_spans, text=new_text)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(normalized_text, self.description, source_data_items=[segment])

        yield normalized_text
