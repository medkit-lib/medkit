from __future__ import annotations

__all__ = [
    "HypothesisDetector",
    "HypothesisDetectorRule",
    "HypothesisRuleMetadata",
    "HypothesisVerbMetadata",
]

import dataclasses
import logging
import re
from pathlib import Path

import yaml
from typing_extensions import Literal, TypedDict

from medkit.core import Attribute
from medkit.core.text import ContextOperation, Segment
from medkit.text.utils.decoding import get_ascii_from_unicode

logger = logging.getLogger(__name__)

_PATH_TO_DEFAULT_RULES = Path(__file__).parent / "hypothesis_detector_default_rules.yml"
_PATH_TO_DEFAULT_VERBS = Path(__file__).parent / "hypothesis_detector_default_verbs.yml"


@dataclasses.dataclass
class HypothesisDetectorRule:
    """Regexp-based rule to use with `HypothesisDetector`

    Attributes
    ----------
    regexp : str
        The regexp pattern used to match a hypothesis
    exclusion_regexps : list of str, optional
        Optional exclusion patterns
    id : str, optional
        Unique identifier of the rule to store in the metadata of the entities
    case_sensitive : bool, default=False
        Whether to ignore case when running `regexp and `exclusion_regexps`
    unicode_sensitive : bool, default=False
        Whether to replace all non-ASCII chars by the closest ASCII chars
        on input text before running `regexp and `exclusion_regexps`.
        If True, then `regexp and `exclusion_regexps` shouldn't contain
        non-ASCII chars because they would never be matched.
    """

    regexp: str
    exclusion_regexps: list[str] = dataclasses.field(default_factory=list)
    id: str | None = None
    case_sensitive: bool = False
    unicode_sensitive: bool = False

    def __post_init__(self):
        assert self.unicode_sensitive or (
            self.regexp.isascii() and all(r.isascii() for r in self.exclusion_regexps)
        ), "HypothesisDetectorRule regexps shouldn't contain non-ASCII chars when unicode_sensitive is False"


class HypothesisRuleMetadata(TypedDict):
    """Metadata dict added to hypothesis attributes with `True` value
    detected by a rule

    Parameters
    ----------
    type : str
        Metadata type, here `"rule"` (use to differentiate
        between rule/verb metadata dict)
    rule_id : str
        Identifier of the rule used to detect an hypothesis.
        If the rule has no uid, then the index of the rule in
        the list of rules is used instead
    """

    type: Literal["rule"]
    rule_id: str


class HypothesisVerbMetadata(TypedDict):
    """Metadata dict added to hypothesis attributes with `True` value
    detected by a rule.

    Parameters
    ----------
    type : str
        Metadata type, here `"verb"` (use to differentiate
        between rule/verb metadata dict).
    matched_verb : str
        Root of the verb used to detect an hypothesis.
    """

    type: Literal["verb"]
    matched_verb: str


class HypothesisDetector(ContextOperation):
    """Annotator creating hypothesis Attributes with boolean values indicating
    if an hypothesis has been found.

    Hypothesis will be considered present either because of the presence of a
    certain text pattern in a segment, or because of the usage of a certain verb
    at a specific mode and tense (for instance conditional).

    Because hypothesis attributes will be attached to whole segments,
    each input segment should be "local"-enough (ie a sentence or a syntagma)
    rather than a big chunk of text.
    """

    def __init__(
        self,
        output_label: str = "hypothesis",
        rules: list[HypothesisDetectorRule] | None = None,
        verbs: dict[str, dict[str, dict[str, list[str]]]] | None = None,
        modes_and_tenses: list[tuple[str, str]] | None = None,
        max_length: int = 150,
        uid: str | None = None,
    ):
        """Instantiate the hypothesis detector

        Parameters
        ----------
        output_label : str, default="hypothesis"
            The label of the created attributes
        rules : list of HypothesisDetectorRule, optional
            The set of rules to use when detecting hypothesis. If none provided,
            the rules in "hypothesis_detector_default_rules.yml" will be used
        verbs : dict of str to dict, optional
            Conjugated verbs forms, to be used in association with `modes_and_tenses`.
            Conjugated forms of a verb at a specific mode and tense must be provided
            in nested dicts with the 1st key being the verb's root, the 2d key the mode
            and the 3d key the tense.
            For instance verb["aller"]["indicatif]["présent"] would hold the list
            ["vais", "vas", "va", "allons", aller", "vont"]
            When `verbs` is provided, `modes_and_tenses` must also be provided.
            If none provided, the rules in "hypothesis_detector_default_verbs.yml" will
            be used.
        modes_and_tenses : list of tuple of str, optional
            List of tuples of all modes and tenses associated with hypothesis.
            Will be used to select conjugated forms in `verbs` that denote hypothesis.
        max_length : int, default=150
            Maximum number of characters in a hypothesis segment. Segments longer than
            this will never be considered as hypothesis
        uid : str, optional
            Identifier of the detector
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if (verbs is None) != (modes_and_tenses is None):
            msg = "'verbs' and 'modes_and_tenses' must be either both provided or both left empty"
            raise ValueError(msg)

        if rules is None:
            rules = self.load_rules(_PATH_TO_DEFAULT_RULES, encoding="utf-8")

        if verbs is None:
            verbs = self.load_verbs(_PATH_TO_DEFAULT_VERBS, encoding="utf-8")
            modes_and_tenses = [
                ("conditionnel", "présent"),
                ("indicatif", "futur simple"),
            ]

        self.check_rules_sanity(rules)

        self.output_label: str = output_label
        self.rules: list[HypothesisDetectorRule] = rules
        self.verbs: dict[str, dict[str, dict[str, list[str]]]] = verbs
        self.modes_and_tenses: list[tuple[str, str]] = modes_and_tenses
        self.max_length: int = max_length

        # build and pre-compile exclusion pattern for each verb
        self._patterns_by_verb = {}
        for verb_root, verb_forms_by_mode_and_tense in verbs.items():
            verb_regexps = set()
            for mode, tense in modes_and_tenses:
                for verb_form in verb_forms_by_mode_and_tense[mode][tense]:
                    verb_regexp = r"\b" + verb_form.replace(" ", r"\s+") + r"\b"
                    verb_regexps.add(verb_regexp)
            verb_pattern = re.compile("|".join(verb_regexps), flags=re.IGNORECASE)
            self._patterns_by_verb[verb_root] = verb_pattern

        # pre-compile rule patterns
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
        """Add an hypothesis attribute to each segment with a boolean value
        indicating if an hypothesis has been detected.

        Hypothesis attributes with a `True` value have a metadata dict with
        fields described in either :class:`.HypothesisRuleMetadata` or :class:`.HypothesisVerbMetadata`.

        Parameters
        ----------
        segments : list of Segment
            List of segments to detect as being hypothesis or not
        """
        for segment in segments:
            hyp_attr = self._detect_hypothesis_in_segment(segment)
            if hyp_attr is not None:
                segment.attrs.add(hyp_attr)

    def _detect_hypothesis_in_segment(self, segment: Segment) -> Attribute | None:
        matched_verb = None
        rule_id = None

        text = segment.text
        # skip empty segments or segments too long
        if len(text) <= self.max_length and self._non_empty_text_pattern.search(text) is not None:
            # match by verb first
            matched_verb = self._find_matching_verb(segment.text)
            # then match by rule if no verb match
            if not matched_verb:
                rule_id = self._find_matching_rule(segment.text)

        if matched_verb is not None:
            hyp_attr = Attribute(
                label=self.output_label,
                value=True,
                metadata=HypothesisVerbMetadata(type="verb", matched_verb=matched_verb),
            )
        elif rule_id is not None:
            hyp_attr = Attribute(
                label=self.output_label,
                value=True,
                metadata=HypothesisRuleMetadata(type="rule", rule_id=rule_id),
            )
        else:
            hyp_attr = Attribute(label=self.output_label, value=False)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(hyp_attr, self.description, source_data_items=[segment])

        return hyp_attr

    def _find_matching_verb(self, text: str) -> str | None:
        for verb, verb_pattern in self._patterns_by_verb.items():
            if verb_pattern.search(text):
                return verb
        return None

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
                # return the rule uid or the rule index if no uid has been set
                return rule.id if rule.id is not None else rule_index

        return None

    @staticmethod
    def load_verbs(path_to_verbs: Path, encoding: str | None = None) -> dict[str, dict[str, dict[str, list[str]]]]:
        """Load all conjugated verb forms stored in a yml file.
        Conjugated verb forms at a specific mode and tense must be stored in nested mappings
        with the 1st key being the verb root, the 2d key the mode and the 3d key the tense.

        Parameters
        ----------
        path_to_verbs : Path
            Path to a yml file containing a list of verbs form,
            arranged by mode and tense.
        encoding : str, optional
            Encoding on the file to open

        Returns
        -------
        dict of str to dict
            List of verb forms in `path_to_verbs`,
            can be used to init an `HypothesisDetector`
        """
        with Path(path_to_verbs).open(encoding=encoding) as fp:
            return yaml.safe_load(fp)

    @staticmethod
    def load_rules(path_to_rules: Path, encoding: str | None = None) -> list[HypothesisDetectorRule]:
        """Load all rules stored in a yml file

        Parameters
        ----------
        path_to_rules : Path
            Path to a yml file containing a list of mappings
            with the same structure as `HypothesisDetectorRule`
        encoding : str, optional
            Encoding of the file to open

        Returns
        -------
        list of HypothesisDetectorRule
            List of all the rules in `path_to_rules`,
            can be used to init an `HypothesisDetector`
        """
        with Path(path_to_rules).open(encoding=encoding) as fp:
            rules_data = yaml.safe_load(fp)
        return [HypothesisDetectorRule(**d) for d in rules_data]

    @classmethod
    def get_example(cls) -> HypothesisDetector:
        """Instantiate an HypothesisDetector with example rules and verbs,
        designed for usage with EDS documents
        """
        rules = cls.load_rules(_PATH_TO_DEFAULT_RULES, encoding="utf-8")
        verbs = cls.load_verbs(_PATH_TO_DEFAULT_VERBS, encoding="utf-8")
        modes_and_tenses = [
            ("conditionnel", "présent"),
            ("indicatif", "futur simple"),
        ]
        return cls(rules=rules, verbs=verbs, modes_and_tenses=modes_and_tenses)

    @staticmethod
    def check_rules_sanity(rules: list[HypothesisDetectorRule]):
        """Check consistency of a set of rules"""
        if any(r.id is not None for r in rules):
            if not all(r.id is not None for r in rules):
                msg = "Some rules have ids and other do not. Please provide either ids for all rules or no ids at all"
                raise ValueError(msg)
            if len({r.id for r in rules}) != len(rules):
                msg = "Some rules have the same uid, each rule must have a unique uid"
                raise ValueError(msg)

    @staticmethod
    def save_rules(
        rules: list[HypothesisDetectorRule],
        path_to_rules: Path,
        encoding: str | None = None,
    ):
        """Store rules in a yml file

        Parameters
        ----------
        rules : list of HypothesisDetectorRule
            The rules to save
        path_to_rules : Path
            Path to a .yml file that will contain the rules
        encoding : str, optional
            Encoding of the .yml file
        """
        with Path(path_to_rules).open(mode="w", encoding=encoding) as fp:
            rules_data = [dataclasses.asdict(r) for r in rules]
            yaml.safe_dump(rules_data, fp)
