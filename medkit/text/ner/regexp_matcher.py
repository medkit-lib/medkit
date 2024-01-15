from __future__ import annotations

__all__ = [
    "RegexpMatcher",
    "RegexpMatcherRule",
    "RegexpMatcherNormalization",
    "RegexpMetadata",
]

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import yaml
from typing_extensions import TypedDict

from medkit.core.text import (
    Entity,
    EntityNormAttribute,
    NEROperation,
    Segment,
    UMLSNormAttribute,
    span_utils,
)
from medkit.text.utils.decoding import get_ascii_from_unicode

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RegexpMatcherRule:
    """Regexp-based rule to use with `RegexpMatcher`

    Attributes
    ----------
    regexp: str
        The regexp pattern used to match entities
    label: str
        The label to attribute to entities created based on this rule
    term: str, optional
        The optional normalized version of the entity text
    id: str, optional
        Unique identifier of the rule to store in the metadata of the entities
    version: str, optional
        Version string to store in the metadata of the entities
    index_extract: int, default=0
        If the regexp has groups, the index of the group to use to extract
        the entity
    case_sensitive: bool, default=True
        Whether to ignore case when running `regexp and `exclusion_regexp`
    unicode_sensitive: bool, default=True
        If True, regexp rule matches are searched on unicode text.
        So, `regexp and `exclusion_regexps` shall not contain non-ASCII chars because
        they would never be matched.
        If False, regexp rule matches are searched on closest ASCII text when possible.
        (cf. RegexpMatcher)
    exclusion_regexp: str, optional
        An optional exclusion pattern. Note that this exclusion pattern will be
        executed on the whole input annotation, so when relying on `exclusion_regexp`
        make sure the input annotations passed to `RegexpMatcher` are "local"-enough
        (sentences or syntagmas) rather than the whole text or paragraphs
    normalizations: list of RegexpMatcherNormalization, optional
        Optional list of normalization attributes that should be attached to
        the entities created
    """

    regexp: str
    label: str
    term: str | None = None
    id: str | None = None
    version: str | None = None
    index_extract: int = 0
    case_sensitive: bool = True
    unicode_sensitive: bool = True
    exclusion_regexp: str | None = None
    normalizations: list[RegexpMatcherNormalization] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert self.unicode_sensitive or (
            self.regexp.isascii() and (self.exclusion_regexp is None or self.exclusion_regexp.isascii())
        ), "RegexpMatcherRule regexps shouldn't contain non-ASCII chars when unicode_sensitive is False"


@dataclasses.dataclass
class RegexpMatcherNormalization:
    """Descriptor of normalization attributes to attach to entities
    created from a `RegexpMatcherRule`

    Attributes
    ----------
    kb_name: str
        The name of the knowledge base we are referencing. Ex: "umls"
    kb_version: str
        The name of the knowledge base we are referencing. Ex: "202AB"
    kb_id: str, optional
        The id of the entity in the knowledge base, for instance a CUI
    """

    kb_name: str
    kb_id: Any
    kb_version: str | None = None


class RegexpMetadata(TypedDict):
    """Metadata dict added to entities matched by :class:`.RegexpMatcher`

    Parameters
    ----------
    rule_id: str or int
        Identifier of the rule used to match an entity.
        If the rule has no id, then the index of the rule in
        the list of rules is used instead.
    version: str, optional
        Optional version of the rule used to match an entity
    """

    rule_id: str | int
    version: str | None


_PATH_TO_DEFAULT_RULES = Path(__file__).parent / "regexp_matcher_default_rules.yml"


class RegexpMatcher(NEROperation):
    """Entity annotator relying on regexp-based rules

    For detecting entities, the module uses rules that may be sensitive to unicode or
    not. When the rule is not sensitive to unicode, we try to convert unicode chars to
    the closest ascii chars. However, some characters need to be pre-processed before
    (e.g., `n°` -> `number`). So, if the text lengths are different, we fall back on
    initial unicode text for detection even if rule is not unicode-sensitive.
    In this case, a warning is logged for recommending to pre-process data.
    """

    def __init__(
        self,
        rules: list[RegexpMatcherRule] | None = None,
        attrs_to_copy: list[str] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Instantiate the regexp matcher

        Parameters
        ----------
        rules: list of RegexpMatcherRule, optional
            The set of rules to use when matching entities. If none provided,
            the rules in "regexp_matcher_default_rules.yml" will be used.
        attrs_to_copy: list of str, optional
            Labels of the attributes that should be copied from the source segment
            to the created entity. Useful for propagating context attributes
            (negation, antecedent, etc).
        name: str, optional
            Name describing the matcher (defaults to the class name)
        uid: str, optional
            Identifier of the matcher
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if rules is None:
            rules = self.load_rules(_PATH_TO_DEFAULT_RULES, encoding="utf-8")
        if attrs_to_copy is None:
            attrs_to_copy = []

        self.check_rules_sanity(rules)

        self.rules = rules
        self.attrs_to_copy = attrs_to_copy

        # pre-compile patterns
        self._patterns = [
            re.compile(rule.regexp, flags=0 if rule.case_sensitive else re.IGNORECASE) for rule in self.rules
        ]
        self._exclusion_patterns = [
            (
                re.compile(
                    rule.exclusion_regexp,
                    flags=0 if rule.case_sensitive else re.IGNORECASE,
                )
                if rule.exclusion_regexp is not None
                else None
            )
            for rule in self.rules
        ]
        self._has_non_unicode_sensitive_rule = any(not r.unicode_sensitive for r in rules)

    def run(self, segments: list[Segment]) -> list[Entity]:
        """Return entities (with optional normalization attributes) matched in `segments`

        Parameters
        ----------
        segments: list of Segment
            List of segments into which to look for matches

        Returns
        -------
        list of Entity:
            Entities found in `segments` (with optional normalization attributes).
            Entities have a metadata dict with fields described in :class:`.RegexpMetadata`
        """
        return [entity for segment in segments for entity in self._find_matches_in_segment(segment)]

    def _find_matches_in_segment(self, segment: Segment) -> Iterator[Entity]:
        text_ascii = None

        if self._has_non_unicode_sensitive_rule:
            text_ascii = get_ascii_from_unicode(segment.text, logger=logger)

        for rule_index in range(len(self.rules)):
            yield from self._find_matches_in_segment_for_rule(rule_index, segment, text_ascii)

    def _find_matches_in_segment_for_rule(
        self, rule_index: int, segment: Segment, text_ascii: str | None
    ) -> Iterator[Entity]:
        rule = self.rules[rule_index]
        pattern = self._patterns[rule_index]
        exclusion_pattern = self._exclusion_patterns[rule_index]

        text_to_match = segment.text if rule.unicode_sensitive else text_ascii

        for match in pattern.finditer(text_to_match):
            # note that we apply exclusion_pattern to the whole segment,
            # so we might have a match in a part of the text unrelated to the current
            # match
            # we could check if we have any exclude match overlapping with
            # the current match but that wouldn't work for all cases
            if exclusion_pattern is not None and exclusion_pattern.search(text_to_match) is not None:
                continue

            # extract raw span list from regex match range
            text, spans = span_utils.extract(segment.text, segment.spans, [match.span(rule.index_extract)])

            rule_id = rule.id if rule.id is not None else rule_index
            metadata = RegexpMetadata(rule_id=rule_id, version=rule.version)

            entity = Entity(
                label=rule.label,
                text=text,
                spans=spans,
                metadata=metadata,
            )

            for label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=label):
                    copied_attr = attr.copy()
                    entity.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            # create normalization attributes for each normalization descriptor
            # of the rule
            norm_attrs = [self._create_norm_attr(norm) for norm in rule.normalizations]
            for norm_attr in norm_attrs:
                entity.attrs.add(norm_attr)

            # create manual normalization attribute from term
            if rule.term is not None:
                entity.attrs.add(EntityNormAttribute(kb_name="rules", kb_id=None, term=rule.term))

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])
                for norm_attr in norm_attrs:
                    self._prov_tracer.add_prov(norm_attr, self.description, source_data_items=[segment])

            yield entity

    @staticmethod
    def _create_norm_attr(norm: RegexpMatcherNormalization) -> EntityNormAttribute:
        if norm.kb_name == "umls":
            norm_attr = UMLSNormAttribute(cui=norm.kb_id, umls_version=norm.kb_version)
        else:
            norm_attr = EntityNormAttribute(kb_name=norm.kb_name, kb_id=norm.kb_id, kb_version=norm.kb_version)
        return norm_attr

    @staticmethod
    def load_rules(path_to_rules: Path, encoding: str | None = None) -> list[RegexpMatcherRule]:
        """Load all rules stored in a yml file

        Parameters
        ----------
        path_to_rules: Path
            Path to a yml file containing a list of mappings
            with the same structure as `RegexpMatcherRule`
        encoding: str, optional
            Encoding of the file to open

        Returns
        -------
        list of RegexpMatcherRule
            List of all the rules in `path_to_rules`,
            can be used to init a `RegexpMatcher`
        """

        class Loader(yaml.Loader):
            pass

        def construct_mapping(loader, node):
            data = loader.construct_mapping(node)
            if "kb_name" in data:
                if "id" in data:  # keep id for retro-compatibility
                    kb_id = data.pop("id")
                    return RegexpMatcherNormalization(kb_id=kb_id, **data)
                return RegexpMatcherNormalization(**data)
            return RegexpMatcherRule(**data)

        Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

        with Path(path_to_rules).open(encoding=encoding) as fp:
            return yaml.load(fp, Loader=Loader)  # noqa: S506

    @staticmethod
    def check_rules_sanity(rules: list[RegexpMatcherRule]):
        """Check consistency of a set of rules"""
        if any(r.id is not None for r in rules):
            if not all(r.id is not None for r in rules):
                msg = "Some rules have ids and other do not. Please provide either ids for all rules or no ids at all"
                raise ValueError(msg)
            if len({r.id for r in rules}) != len(rules):
                msg = "Some rules have the same id, each rule must have a unique id"
                raise ValueError(msg)

    @staticmethod
    def save_rules(rules: list[RegexpMatcherRule], path_to_rules: Path, encoding: str | None = None):
        """Store rules in a yml file

        Parameters
        ----------
        rules: list of RegexpMatcherRule
            The rules to save
        path_to_rules: Path
            Path to a .yml file that will contain the rules
        encoding: str, optional
            Encoding of the .yml file
        """
        with Path(path_to_rules).open(mode="w", encoding=encoding) as fp:
            rules_data = [dataclasses.asdict(r) for r in rules]
            yaml.safe_dump(rules_data, fp)
