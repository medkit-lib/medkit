from __future__ import annotations

__all__ = ["FamilyDetector", "FamilyDetectorRule", "FamilyMetadata"]

import dataclasses
import logging
import re
from pathlib import Path

import yaml
from typing_extensions import TypedDict

from medkit.core import Attribute
from medkit.core.text import ContextOperation, Segment
from medkit.text.utils.decoding import get_ascii_from_unicode

logger = logging.getLogger(__name__)

_PATH_TO_DEFAULT_RULES = Path(__file__).parent / "family_detector_default_rules.yml"


@dataclasses.dataclass
class FamilyDetectorRule:
    """Regexp-based rule to use with `FamilyDetector`

    Input text may be converted before detecting rule.

    Parameters
    ----------
    regexp : str
        The regexp pattern used to match a family reference
    exclusion_regexps : list of str, optional
        Optional exclusion patterns
    id : str, optional
        Unique identifier of the rule to store in the metadata of the entities
    case_sensitive : bool, default=False
        Whether to consider case when running `regexp and `exclusion_regexs`
    unicode_sensitive : bool, default=False
        If True, rule matches are searched on unicode text.
        So, `regexp` and `exclusion_regexps` shall not contain non-ASCII chars because
        they would never be matched.
        If False, rule matches are searched on closest ASCII text when possible.
        (cf. FamilyDetector)
    """

    regexp: str
    exclusion_regexps: list[str] = dataclasses.field(default_factory=list)
    id: str | None = None
    case_sensitive: bool = False
    unicode_sensitive: bool = False

    def __post_init__(self):
        assert self.unicode_sensitive or (
            self.regexp.isascii() and all(r.isascii() for r in self.exclusion_regexps)
        ), "FamilyDetectorRule regexps shouldn't contain non-ASCII chars when unicode_sensitive is False"


class FamilyMetadata(TypedDict):
    """Metadata dict added to family attributes with `True` value.

    Parameters
    ----------
    rule_id : str or int
        Identifier of the rule used to detect a family reference.
        If the rule has no id, then the index of the rule in
        the list of rules is used instead.
    """

    rule_id: str | int


class FamilyDetector(ContextOperation):
    """Annotator creating family attributes with boolean values
    indicating if a family reference has been detected.

    Because family attributes will be attached to whole annotations,
    each input annotation should be "local"-enough rather than
    a big chunk of text (ie a sentence or a syntagma).

    For detecting family references, the module uses rules that may be sensitive to unicode or
    not. When the rule is not sensitive to unicode, we try to convert unicode chars to
    the closest ascii chars. However, some characters need to be pre-processed before
    (e.g., `n°` -> `number`). So, if the text lengths are different, we fall back on
    initial unicode text for detection even if rule is not unicode-sensitive.
    In this case, a warning is logged for recommending to pre-process data.

    Note that for better results, family detection should be run at the sentence
    level (ie on sentence segments) rather than at the syntagma level [1].

    [1] N. Garcelon, A. Neuraz, V. Benoit, R. Salomon, A. Burgun, "Improving a
    full-text search engine: the importance of negation detection and family
    history context to identify cases in a biomedical data warehouse", Journal
    of the American Medical Informatics Association, Volume 24, Issue 3, May
    2017
    """

    def __init__(
        self,
        output_label: str,
        rules: list[FamilyDetectorRule] | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        output_label : str
            The label of the created attributes
        rules : list of FamilyDetectorRule, optional
            The set of rules to use when detecting family references. If none provided,
            the rules in "family_detector_default_rules.yml" will be used
        uid : str, optional
            Identifier of the detector
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if rules is None:
            rules = self.load_rules(_PATH_TO_DEFAULT_RULES, encoding="utf-8")

        self.check_rules_sanity(rules)

        self.output_label = output_label
        self.rules = rules

        # pre-compile patterns
        self._non_empty_text_pattern = re.compile(r"[a-z]", flags=re.IGNORECASE)
        self._patterns = [
            re.compile(rule.regexp, flags=0 if rule.case_sensitive else re.IGNORECASE) for rule in self.rules
        ]
        self._exclusion_patterns = [
            (
                re.compile(
                    "|".join(f"(?:{r})" for r in rule.exclusion_regexps),  # join all exclusions in one pattern
                    flags=0 if rule.case_sensitive else re.IGNORECASE,
                )
                if rule.exclusion_regexps
                else None
            )
            for rule in self.rules
        ]
        self._has_non_unicode_sensitive_rule = any(not r.unicode_sensitive for r in rules)

    def run(self, segments: list[Segment]):
        """Add a family attribute to each segment with a boolean value
        indicating if a family reference has been detected.

        Family attributes with a `True` value have a metadata dict with
        fields described in :class:`.FamilyMetadata`.

        Parameters
        ----------
        segments : list of Segment
            List of segments to detect as being family references or not
        """
        for segment in segments:
            family_attr = self._detect_family_ref_in_segment(segment)
            if family_attr is not None:
                segment.attrs.add(family_attr)

    def _detect_family_ref_in_segment(self, segment: Segment) -> Attribute | None:
        rule_id = self._find_matching_rule(segment.text)
        if rule_id is not None:
            family_attr = Attribute(
                label=self.output_label,
                value=True,
                metadata=FamilyMetadata(rule_id=rule_id),
            )
        else:
            family_attr = Attribute(label=self.output_label, value=False)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(family_attr, self.description, source_data_items=[segment])

        return family_attr

    def _find_matching_rule(self, text: str) -> str | int | None:
        # skip empty text
        if self._non_empty_text_pattern.search(text) is None:
            return None

        text_unicode = text
        text_ascii = None

        if self._has_non_unicode_sensitive_rule:
            text_ascii = get_ascii_from_unicode(text, logger=logger)

        # try all rules until we have a match
        for rule_index, rule in enumerate(self.rules):
            pattern = self._patterns[rule_index]
            exclusion_pattern = self._exclusion_patterns[rule_index]
            text = text_unicode if rule.unicode_sensitive else text_ascii
            if pattern.search(text) and not (exclusion_pattern and exclusion_pattern.search(text)):
                # return the rule id or the rule index if no id has been set
                return rule.id if rule.id is not None else rule_index

        return None

    @staticmethod
    def load_rules(path_to_rules: Path, encoding: str | None = None) -> list[FamilyDetectorRule]:
        """Load all rules stored in a yml file

        Parameters
        ----------
        path_to_rules : Path
            Path to a yml file containing a list of mappings
            with the same structure as `FamilyDetectorRule`
        encoding : str, optional
            Encoding of the file to open

        Returns
        -------
        list of FamilyDetectorRule
            List of all the rules in `path_to_rules`,
            can be used to init a `FamilyDetector`
        """
        with Path(path_to_rules).open(encoding=encoding) as fp:
            rules_data = yaml.safe_load(fp)
        return [FamilyDetectorRule(**d) for d in rules_data]

    @staticmethod
    def check_rules_sanity(rules: list[FamilyDetectorRule]):
        """Check consistency of a set of rules"""
        if any(r.id is not None for r in rules):
            if not all(r.id is not None for r in rules):
                msg = "Some rules have ids and other do not. Please provide either ids for all rules or no ids at all"
                raise ValueError(msg)
            if len({r.id for r in rules}) != len(rules):
                msg = "Some rules have the same id, each rule must have a unique id"
                raise ValueError(msg)

    @staticmethod
    def save_rules(
        rules: list[FamilyDetectorRule],
        path_to_rules: Path,
        encoding: str | None = None,
    ):
        """Store rules in a yml file

        Parameters
        ----------
        rules : list of FamilyDetectorRule
            The rules to save
        path_to_rules : Path
            Path to a .yml file that will contain the rules
        encoding : str, optional
            Encoding of the .yml file
        """
        with Path(path_to_rules).open(mode="w", encoding=encoding) as fp:
            rules_data = [dataclasses.asdict(r) for r in rules]
            yaml.safe_dump(rules_data, fp)
