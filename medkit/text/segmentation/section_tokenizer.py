from __future__ import annotations

__all__ = ["SectionModificationRule", "SectionTokenizer"]

import dataclasses
from pathlib import Path
from typing import Iterable

import yaml
from flashtext import KeywordProcessor
from typing_extensions import Literal

from medkit.core import Attribute
from medkit.core.text import Segment, SegmentationOperation, span_utils
from medkit.core.text.utils import lstrip, rstrip

_PATH_TO_DEFAULT_RULES = Path(__file__).parent / "default_section_definition.yml"


@dataclasses.dataclass
class SectionModificationRule:
    section_name: str
    new_section_name: str
    other_sections: list[str]
    order: Literal["BEFORE", "AFTER"]


class SectionTokenizer(SegmentationOperation):
    """Section segmentation annotator based on keyword rules"""

    _DEFAULT_LABEL: str = "section"
    _DEFAULT_STRIP_CHARS: str = ".;,?! \n\r\t"

    def __init__(
        self,
        section_dict: dict[str, list[str]] | None = None,
        output_label: str = _DEFAULT_LABEL,
        section_rules: Iterable[SectionModificationRule] = (),
        strip_chars: str = _DEFAULT_STRIP_CHARS,
        uid: str | None = None,
    ):
        """Initialize the Section Tokenizer

        Parameters
        ----------
        section_dict: dict of str to list of str, optional
            Dictionary containing the section name as key and the list of mappings as
            value. If None, the content of default_section_definition.yml will be used.
        output_label: str, optional
            Segment label to use for annotation output.
        section_rules: iterable of SectionModificationRule, optional
            List of rules for modifying a section name according its order to the other
            sections. If section_dict is None, the content of
            default_section_definition.yml will be used.
        strip_chars: str, optional
            The list of characters to strip at the beginning of the returned segment.
        uid: str, optional
            Identifier of the tokenizer
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        self.strip_chars = strip_chars

        if section_dict is None:
            section_dict, section_rules = self.load_section_definition(_PATH_TO_DEFAULT_RULES, encoding="utf-8")

        self.section_dict = section_dict
        self.section_rules = tuple(section_rules)

        self.keyword_processor = KeywordProcessor(case_sensitive=True)
        self.keyword_processor.add_keywords_from_dict(section_dict)

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Return sections detected in `segments`.
        Each section is a segment with an attached attribute
        (label: <same as self.output_label>, value: <the name of the section>).

        Parameters
        ----------
        segments: list of Segment
            List of segments into which to look for sections

        Returns
        -------
        list of Segment
            Sections segments found in `segments`
        """
        return [section for segment in segments for section in self._find_sections_in_segment(segment)]

    def _find_sections_in_segment(self, segment: Segment):
        # Process mappings
        match = self.keyword_processor.extract_keywords(segment.text, span_info=True)

        # Sort according to the match start
        match.sort(key=lambda x: x[1])
        if len(match) == 0 or match[0][1] != 0:
            # Add head before any detected sections
            match.insert(0, ("head", 0, 0))

        # Get sections to rename according defined rules
        # e.g., set any 'traitement' section occurring before 'histoire' or 'evolution'
        # to 'traitement entree' (cf. example)
        new_sections = self._get_sections_to_rename(match)

        for index, section in enumerate(match):
            name = new_sections.get(index, section[0])
            if index != len(match) - 1:
                ranges = [(section[1], match[index + 1][1])]
            else:
                ranges = [(section[1], len(segment.text))]

            # Remove extra characters at beginning of the detected segments
            # and white spaces at end of the text
            strip_ranges = []
            for start, end in ranges:
                text, new_start = lstrip(segment.text[start:end], start, self.strip_chars)
                text, new_end = rstrip(text, end)
                if len(text) == 0:  # empty segment
                    continue
                strip_ranges.append((new_start, new_end))

            # Extract medkit spans from relative spans (i.e., ranges)
            text, spans = span_utils.extract(
                text=segment.text,
                spans=segment.spans,
                ranges=strip_ranges,
            )

            # add section name in metadata
            metadata = {"name": name}
            new_section = Segment(
                label=self.output_label,
                spans=spans,
                text=text,
                metadata=metadata,
            )

            # add section name in section attribute
            attr = Attribute(label=self.output_label, value=name)
            new_section.attrs.add(attr)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_section, self.description, source_data_items=[segment])
                self._prov_tracer.add_prov(attr, self.description, source_data_items=[segment])

            yield new_section

    def _get_sections_to_rename(self, match: list[tuple]):
        match_type = [m[0] for m in match]
        map_index_new_name = {}
        list_to_process = ()
        for rule in self.section_rules:
            if rule.order == "BEFORE":
                # Change section name if section is before one of the listed sections
                list_to_process = enumerate(match_type)
            elif rule.order == "AFTER":
                # Change section name if the section is after one of the listed sections
                list_to_process = reversed(list(enumerate(match_type)))

            # Navigate in list according to the order defined above
            candidate_sections = []
            for index, section_name in list_to_process:
                if section_name == rule.section_name:
                    candidate_sections.append(index)
                if section_name in rule.other_sections:
                    for candidate_index in candidate_sections:
                        map_index_new_name[candidate_index] = rule.new_section_name
                    candidate_sections.clear()

        return map_index_new_name

    @classmethod
    def get_example(cls):
        config_path = _PATH_TO_DEFAULT_RULES
        section_dict, section_rules = cls.load_section_definition(config_path, encoding="utf-8")
        return cls(
            section_dict=section_dict,
            section_rules=section_rules,
        )

    @staticmethod
    def load_section_definition(
        filepath: Path, encoding: str | None = None
    ) -> tuple[dict[str, list[str]], tuple[SectionModificationRule, ...]]:
        """Load the sections definition stored in a yml file

        Parameters
        ----------
        filepath : Path
            Path to a yml file containing the sections(name + mappings) and rules
        encoding : str, optional
            Encoding of the file to open

        Returns
        -------
        tuple
            Tuple containing:
            - the dictionary where key is the section name and value is the list of all
            equivalent strings.
            - the list of section modification rules.
            These rules allow to rename some sections according their order
        """
        with Path(filepath).open(encoding=encoding) as fp:
            config = yaml.safe_load(fp)

        section_dict = config["sections"]
        section_rules = tuple(SectionModificationRule(**rule) for rule in config["rules"])

        return section_dict, section_rules

    @staticmethod
    def save_section_definition(
        section_dict: dict[str, list[str]],
        section_rules: Iterable[SectionModificationRule],
        filepath: Path,
        encoding: str | None = None,
    ):
        """Save section yaml definition file

        Parameters
        ----------
        section_dict : dict of str to list of str
            Dictionary containing the section name as key and the list of mappings
            as value (cf. content of default_section_dict.yml as example)
        section_rules : iterable of SectionModificationRule
            List of rules for modifying a section name according its order to the other
            sections.
        filepath : Path
            Path to the file to save
        encoding : str, optional
            File encoding
        """
        with Path(filepath).open(mode="w", encoding=encoding) as fp:
            data = {"sections": section_dict, "rules": []}
            for rule in section_rules:
                data["rules"].append(dataclasses.asdict(rule))
            yaml.safe_dump(data, fp, allow_unicode=True, encoding=encoding)
