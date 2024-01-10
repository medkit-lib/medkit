from __future__ import annotations

__all__ = ["SimstringMatcher", "SimstringMatcherRule", "SimstringMatcherNormalization"]

import dataclasses
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from typing_extensions import Literal

from medkit.text.ner._base_simstring_matcher import (
    BaseSimstringMatcher,
    BaseSimstringMatcherNormalization,
    BaseSimstringMatcherRule,
    build_simstring_matcher_databases,
)

_RULES_DB_FILENAME = "rules.db"
_SIMSTRING_DB_FILENAME = "simstring"


@dataclasses.dataclass
class SimstringMatcherRule(BaseSimstringMatcherRule):
    """
    Rule to use with :class:`~.SimstringMatcher`

    Attributes
    ----------
    term:
        Term to match using similarity-based fuzzy matching
    label:
        Label to use for the entities created when a match is found
    case_sensitive:
        Whether to take case into account when looking for matches.
    unicode_sensitive:
        Whether to use ASCII-only versions of the rule term and input texts when
        looking for matches (non-ASCII chars replaced by closest ASCII chars).
    normalizations:
        Optional list of normalization attributes that should be attached to the
        entities created
    """

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SimstringMatcherRule:
        """
        Creates a SimStringMatcherRule from a dict.
        """
        return SimstringMatcherRule(
            term=data["term"],
            label=data["label"],
            case_sensitive=data["case_sensitive"],
            unicode_sensitive=data["unicode_sensitive"],
            normalizations=[SimstringMatcherNormalization.from_dict(n) for n in data["normalizations"]],
        )


class SimstringMatcherNormalization(BaseSimstringMatcherNormalization):
    """
    Descriptor of normalization attributes to attach to entities
    created from a :class:`~.SimstringMatcherRule`

    Attributes
    ----------
    kb_name:
        The name of the knowledge base we are referencing. Ex: "umls"
    kb_version:
        The name of the knowledge base we are referencing. Ex: "202AB"
    kb_id:
        The id of the entity in the knowledge base, for instance a CUI
    term:
        Optional normalized version of the entity text in the knowledge base
    """

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SimstringMatcherNormalization:
        """Creates a SimstringMatcherNormalization object from a dict"""
        return SimstringMatcherNormalization(
            kb_name=data["kb_name"],
            kb_version=data["kb_version"],
            kb_id=data["id"],
            term=data["term"],
        )


class SimstringMatcher(BaseSimstringMatcher):
    """
    Entity matcher relying on string similarity

    Uses the `simstring` fuzzy matching algorithm
    (http://chokkan.org/software/simstring/).

    Note that setting `spacy_tokenization_language` to `True` might reduce the
    number of false positives. This requires the `spacy` optional dependency,
    which can be installed with `pip install medkit-lib[spacy]`.
    """

    def __init__(
        self,
        rules: List[SimstringMatcherRule],
        threshold: float = 0.9,
        min_length: int = 3,
        max_length: int = 50,
        similarity: Literal["cosine", "dice", "jaccard", "overlap"] = "jaccard",
        spacy_tokenization_language: Optional[str] = None,
        blacklist: Optional[List[str]] = None,
        same_beginning: bool = False,
        attrs_to_copy: Optional[List[str]] = None,
        name: Optional[str] = None,
        uid: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        rules:
            Rules to use for matching entities.
        min_length:
            Minimum number of chars in matched entities.
        max_length:
            Maximum number of chars in matched entities.
        threshold:
            Minimum similarity (between 0.0 and 1.0) between a rule term and the
            text of an entity matched on that rule.
        similarity:
            Similarity metric to use.
        spacy_tokenization_language:
            2-letter code (ex: "fr", "en", etc.) designating the language of the
            spacy model to use for tokenization. If provided, spacy will be used
            to tokenize input segments and filter out some tokens based on their
            part-of-speech tags, such as determinants, conjunctions and
            prepositions. If `None`, a simple regexp based tokenization will be
            used, which is faster but might give more false positives.
        blacklist:
            Optional list of exact terms to ignore.
        same_beginning:
            Ignore all matches that start with a different character than the
            term of the rule. This can be convenient to get rid of false
            positives on words that are very similar but have opposite meanings
            because of a preposition, for instance "activation" and
            "inactivation".
        attrs_to_copy:
            Labels of the attributes that should be copied from the source
            segment to the created entity. Useful for propagating context
            attributes (negation, antecedent, etc.).
        name:
            Name describing the matcher (defaults to the class name).
        uid:
            Identifier of the matcher.
        """

        self._temp_dir = tempfile.TemporaryDirectory()
        rules_db_file = Path(self._temp_dir.name) / _RULES_DB_FILENAME
        simstring_db_file = Path(self._temp_dir.name) / _SIMSTRING_DB_FILENAME

        build_simstring_matcher_databases(
            simstring_db_file,
            rules_db_file,
            rules,
        )

        super().__init__(
            simstring_db_file=simstring_db_file,
            rules_db_file=rules_db_file,
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
            similarity=similarity,
            spacy_tokenization_language=spacy_tokenization_language,
            blacklist=blacklist,
            same_beginning=same_beginning,
            attrs_to_copy=attrs_to_copy,
            name=name,
            uid=uid,
        )

    @staticmethod
    def load_rules(path_to_rules: Path, encoding: Optional[str] = None) -> List[SimstringMatcherRule]:
        """
        Load all rules stored in a yml file

        Parameters
        ----------
        path_to_rules
            The path to a yml file containing a list of mappings with the same
            structure as :class:`~.SimstringMatcherRule`
        encoding
            The encoding of the file to open

        Returns
        -------
        List[SimstringMatcherRule]
            List of all the rules in `path_to_rules`, can be used to init a
            :class:`~.SimstringMatcher`
        """

        class _Loader(yaml.Loader):
            pass

        def _construct_mapping(loader, node):
            data = loader.construct_mapping(node)
            if "kb_name" in data:
                return SimstringMatcherNormalization(**data)
            else:
                return SimstringMatcherRule(**data)

        _Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

        with open(path_to_rules, encoding=encoding) as f:
            rules = yaml.load(f, Loader=_Loader)
        return rules

    @staticmethod
    def save_rules(
        rules: List[SimstringMatcherRule],
        path_to_rules: Path,
        encoding: Optional[str] = None,
    ):
        """
        Store rules in a yml file

        Parameters
        ----------
        rules
            The rules to save
        path_to_rules
            The path to a yml file that will contain the rules
        encoding
            The encoding of the yml file
        """

        with open(path_to_rules, mode="w", encoding=encoding) as f:
            rules_data = [dataclasses.asdict(r) for r in rules]
            yaml.safe_dump(rules_data, f)
