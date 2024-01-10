from __future__ import annotations

__all__ = ["SyntagmaTokenizer"]

import pathlib
import re
from typing import Iterator, List, Optional, Tuple

import yaml

from medkit.core.text import Segment, SegmentationOperation, span_utils
from medkit.core.text.utils import lstrip, rstrip

_PATH_TO_DEFAULT_RULES = pathlib.Path(__file__).parent / "default_syntagma_definition.yml"


class SyntagmaTokenizer(SegmentationOperation):
    """Syntagma segmentation annotator based on provided separators"""

    _DEFAULT_LABEL = "syntagma"
    _DEFAULT_STRIP_CHARS = ".;,?! \n\r\t"

    def __init__(
        self,
        separators: Tuple[str, ...] = None,
        output_label: str = _DEFAULT_LABEL,
        strip_chars: str = _DEFAULT_STRIP_CHARS,
        attrs_to_copy: Optional[List[str]] = None,
        uid: Optional[str] = None,
    ):
        """
        Instantiate the syntagma tokenizer

        Parameters
        ----------
        separators: Tuple[str, ...]
            The tuple of regular expressions corresponding to separators. If None
            provided, the rules in "default_syntagma_definitiion.yml" will be used.
        output_label: str, Optional
            The output label of the created annotations.
        strip_chars
            The list of characters to strip at the beginning of the returned segment.
        attrs_to_copy:
            Labels of the attributes that should be copied from the input segment
            to the derived segment. For example, useful for propagating section name.
        uid: str, Optional
            Identifier of the tokenizer
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.output_label = output_label
        self.separators = separators
        self.strip_chars = strip_chars
        if separators is None:
            self.separators = self.load_syntagma_definition(_PATH_TO_DEFAULT_RULES, encoding="utf-8")
        self.attrs_to_copy = attrs_to_copy

    def run(self, segments: List[Segment]) -> List[Segment]:
        """
        Return syntagmes detected in `segments`.

        Parameters
        ----------
        segments:
            List of segments into which to look for sentences

        Returns
        -------
        List[Segments]:
            Syntagmas segments found in `segments`
        """
        return [syntagma for segment in segments for syntagma in self._find_syntagmas_in_segment(segment)]

    def _find_syntagmas_in_segment(self, segment: Segment) -> Iterator[Segment]:
        regex_rule = (
            "(?P<blanks> *)"  # Blanks at the beginning of the syntagmas
            + "(?P<syntagma>.+?)"  # Syntagma to detect
            + "(?P<separator>"  # Separator
            + "|".join(self.separators)
            + "|$)"  # including the last syntagma without end separator
        )
        pattern = re.compile(regex_rule, flags=re.DOTALL)

        sep_exists = False
        start = 0

        for match in pattern.finditer(segment.text):
            start = match.start("syntagma") if not sep_exists else start
            end = match.end("syntagma")
            sep_exists = True

            # Remove extra characters at beginning of the detected segments
            # and white spaces at end of the text
            text, start = lstrip(segment.text[start:end], start, self.strip_chars)
            text, end = rstrip(text, end)

            # Ignore empty syntagmas
            if len(text) == 0:
                start = match.start("separator")
                continue

            # Extract raw span list from regex match ranges
            text, spans = span_utils.extract(
                text=segment.text,
                spans=segment.spans,
                ranges=[(start, end)],
            )

            # Give next syntagma start
            start = match.start("separator")

            syntagma = Segment(
                label=self.output_label,
                spans=spans,
                text=text,
            )

            # Copy inherited attributes
            for label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=label):
                    copied_attr = attr.copy()
                    syntagma.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(syntagma, self.description, source_data_items=[segment])

            yield syntagma

    @classmethod
    def get_example(cls):
        config_path = _PATH_TO_DEFAULT_RULES
        separators = cls.load_syntagma_definition(config_path, encoding="utf-8")
        syntagma_tokenizer = cls(separators=separators)
        return syntagma_tokenizer

    @staticmethod
    def load_syntagma_definition(filepath: pathlib.Path, encoding: Optional[str] = None) -> Tuple[str, ...]:
        """
        Load the syntagma definition stored in yml file

        Parameters
        ----------
        filepath:
            Path to a yml file containing the syntagma separators
        encoding
            Encoding of the file to open

        Returns
        -------
        Tuple[str, ...]:
            Tuple containing the separators
        """

        with open(filepath, mode="r", encoding=encoding) as f:
            config = yaml.safe_load(f)

        syntagma_seps = tuple(str(sep) for sep in config["syntagmas"]["separators"])

        return syntagma_seps

    @staticmethod
    def save_syntagma_definition(
        syntagma_seps: Tuple[str, ...],
        filepath: pathlib.Path,
        encoding: Optional[str] = None,
    ):
        """
        Save syntagma yaml definition file

        Parameters
        ----------
        syntagma_seps
            The tuple of regular expressions corresponding to separators
        filepath
            The path of the file to save
        encoding
            The encoding of the file. Default: None
        """
        data = {
            "syntagmas": {"separators": []},
        }
        for sep in syntagma_seps:
            data["syntagmas"]["separators"].append(sep)

        with open(filepath, "w", encoding=encoding) as f:
            yaml.safe_dump(data, f, allow_unicode=True)
