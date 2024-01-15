from __future__ import annotations

__all__ = [
    "BaseSimstringMatcher",
    "BaseSimstringMatcherRule",
    "BaseSimstringMatcherNormalization",
    "build_simstring_matcher_databases",
]

import dataclasses
import math
import re
import shelve
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from anyascii import anyascii
from pysimstring import simstring
from typing_extensions import Literal

try:
    import spacy
except ImportError:
    spacy = None


from medkit.core.text import (
    Entity,
    EntityNormAttribute,
    NEROperation,
    Segment,
    UMLSNormAttribute,
    span_utils,
)

if TYPE_CHECKING:
    from pathlib import Path

_SIMILARITY_MAP = {
    "cosine": simstring.cosine,
    "dice": simstring.dice,
    "jaccard": simstring.jaccard,
    "overlap": simstring.overlap,
}


@dataclasses.dataclass
class BaseSimstringMatcherRule:
    """Rule to use with :class:`~.BaseSimstringMatcher`

    Attributes
    ----------
    term : str
        Term to match using similarity-based fuzzy matching
    label : str
        Label to use for the entities created when a match is found
    case_sensitive : bool, default=False
        Whether to take case into account when looking for matches.
    unicode_sensitive : bool, default=False
        Whether to use ASCII-only versions of the rule term and input texts when
        looking for matches (non-ASCII chars replaced by closest ASCII chars).
    normalizations : list of BaseSimstringMatcherNormalization, optional
        Optional list of normalization attributes that should be attached to the
        entities created
    """

    term: str
    label: str
    case_sensitive: bool = False
    unicode_sensitive: bool = False
    normalizations: list[BaseSimstringMatcherNormalization] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BaseSimstringMatcherNormalization:
    """Descriptor of normalization attributes to attach to entities
    created from a `~.BaseSimstringMatcherRule`

    Attributes
    ----------
    kb_name : str
        The name of the knowledge base we are referencing. Ex: "umls"
    kb_id : int or str
        The id of the entity in the knowledge base, for instance a CUI
    kb_version : str, optional
        The name of the knowledge base we are referencing. Ex: "202AB"
    term : str, optional
        Optional normalized version of the entity text in the knowledge base
    """

    kb_name: str
    kb_id: int | str
    kb_version: str | None = None
    term: str | None = None

    def to_attribute(self: BaseSimstringMatcherNormalization, score: float) -> EntityNormAttribute:
        """Create a normalization attribute based on the normalization descriptor

        Parameters
        ----------
        score : float
            Score of similarity between the normalized term and the entity text

        Returns
        -------
        EntityNormAttribute
            Normalization attribute to add to entity
        """
        if self.kb_name == "umls":
            norm_attr = UMLSNormAttribute(
                cui=self.kb_id,
                umls_version=self.kb_version,
                term=self.term,
                score=score,
            )
        else:
            norm_attr = EntityNormAttribute(
                kb_name=self.kb_name,
                kb_id=self.kb_id,
                kb_version=self.kb_version,
                term=self.term,
                score=score,
            )
        return norm_attr


@dataclasses.dataclass
class _Match:
    start: int
    end: int
    rule: BaseSimstringMatcherRule
    score: float

    @property
    def length(self):
        return self.end - self.start

    def overlaps(self, other: _Match):
        return (other.start <= self.end) and (other.end >= self.start)


class BaseSimstringMatcher(NEROperation):
    """Base class for entity matcher using the `simstring` fuzzy matching algorithm
    (also used by `QuickUMLS`).
    """

    def __init__(
        self,
        simstring_db_file: Path,
        rules_db_file: Path,
        threshold: float = 0.9,
        min_length: int = 3,
        max_length: int = 50,
        similarity: Literal["cosine", "dice", "jaccard", "overlap"] = "jaccard",
        spacy_tokenization_language: str | None = None,
        blacklist: list[str] | None = None,
        same_beginning: bool = False,
        attrs_to_copy: list[str] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        simstring_db_file : Path
            Simstring database to use
        rules_db_file : Path
            Rules database (in python shelve format) mapping matched terms to
            corresponding rules
        threshold : float, default=0.9
            Minimum similarity (between 0.0 and 1.0) between a rule term and the
            text of an entity matched on that rule.
        min_length : int, default=3
            Minimum number of chars in matched entities.
        max_length : int, default=50
            Maximum number of chars in matched entities.
        similarity : str, default="jaccard"
            Similarity metric to use.
        spacy_tokenization_language : str, optional
            2-letter code (ex: "fr", "en", etc) designating the language of the
            spacy model to use for tokenization. If provided, spacy will be used
            to tokenize input segments and filter out some tokens based on their
            part-of-speech tags, such as determinants, conjunctions and
            prepositions. If `None`, a simple regexp based tokenization will be
            used, which is faster but might give more false positives.
        blacklist : list of str, optional
            Optional list of exact terms to ignore.
        same_beginning : bool, default=False
            Ignore all matches that start with a different character than the
            term of the rule. This can be convenient to get rid of false
            positives on words that are very similar but have opposite meanings
            because of a preposition, for instance "activation" and
            "inactivation".
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the source
            segment to the created entity. Useful for propagating context
            attributes (negation, antecedent, etc).
        name : str, optional
            Name describing the matcher (defaults to the class name).
        uid : str, optional
            Identifier of the matcher.
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        assert (
            similarity in _SIMILARITY_MAP
        ), f"Invalid similarity '{similarity}', must be one of {list(_SIMILARITY_MAP.keys())}"

        if blacklist is None:
            blacklist = []
        if attrs_to_copy is None:
            attrs_to_copy = []

        self.min_length = min_length
        self.max_length = max_length
        self.threshold = threshold
        self.similarity = similarity
        self.blacklist = set(blacklist)
        self.same_beginning = same_beginning
        self.attrs_to_copy = attrs_to_copy

        self._simstring_db_reader = simstring.reader(str(simstring_db_file))
        self._simstring_db_reader.measure = _SIMILARITY_MAP[similarity]
        self._simstring_db_reader.threshold = threshold

        self._rules_db = shelve.open(str(rules_db_file), flag="r")  # noqa: S301

        if spacy_tokenization_language is not None:
            if not spacy:
                msg = "Spacy module must be installed to use the 'spacy_language_code' init parameter"
                raise ModuleNotFoundError(msg)
            if spacy_tokenization_language == "en":
                spacy_model = "en_core_web_sm"
            else:
                spacy_model = f"{spacy_tokenization_language}_core_news_sm"
            self._spacy_lang = spacy.load(
                spacy_model,
                # only keep tok2vec and morphologizer to get POS tags
                disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
            )
        else:
            self._spacy_lang = None

    def run(self, segments: list[Segment]) -> list[Entity]:
        """Return entities (with optional normalization attributes) matched in `segments`

        Parameters
        ----------
        segments : list of Segment
            List of segments into which to look for matches

        Returns
        -------
        list of Entity
            Entities found in `segments` (with optional normalization
            attributes)
        """
        # pre-tokenize all segments with pipe() so spacy can parallelize it
        spacy_docs = self._spacy_lang.pipe(s.text for s in segments) if self._spacy_lang else [None] * len(segments)

        return [
            entity
            for segment, spacy_doc in zip(segments, spacy_docs)
            for entity in self._find_matches_in_segment(segment, spacy_doc)
        ]

    def _find_matches_in_segment(self, segment: Segment, spacy_doc: Any | None) -> Iterator[Entity]:
        """Return an iterator to the entities matched in a segment"""
        text = segment.text
        matches = []
        if spacy_doc is not None:
            ranges = _build_candidate_ranges_with_spacy(spacy_doc, self.min_length, self.max_length)
        else:
            ranges = _build_candidate_ranges_with_regexp(text, self.min_length, self.max_length)

        for start, end in ranges:
            candidate_text = text[start:end]
            # simstring matching is always performed on lowercased ASCII-only text,
            # then for potential matches we will recompute the similarity
            # taking into account the actual rule parameters
            candidate_text_processed = anyascii(candidate_text.lower())
            matched_terms = self._simstring_db_reader.retrieve(candidate_text_processed)

            for matched_term in matched_terms:
                # if requested, ignore matches that start differently
                if self.same_beginning and matched_term[0] != candidate_text_processed[0]:
                    continue

                # retrieve rules corresponding to matched term
                rules = self._rules_db[matched_term]
                # for each rule, recompute similarity
                # taking into account case_sensitivity and unicode_sensitivity
                for rule in rules:
                    rule_term = rule.term

                    # apply required text transforms on term and candidate texts
                    if not rule.case_sensitive and not rule.unicode_sensitive:
                        candidate_text = candidate_text_processed
                        rule_term = matched_term
                    elif not rule.case_sensitive:
                        candidate_text = candidate_text.lower()
                        rule_term = rule_term.lower()
                    elif not rule.unicode_sensitive:
                        candidate_text = anyascii(candidate_text)
                        rule_term = anyascii(rule_term)

                    # ignore blacklisted terms
                    if rule_term in self.blacklist:
                        continue

                    # recompute similarity and keep match if above threshold
                    score = _get_similarity_score(rule_term, candidate_text, self.similarity)
                    if score >= self.threshold:
                        match = _Match(start, end, rule, score)
                        matches.append(match)

        # keep only best matches among overlaps
        matches = self._filter_overlapping_matches(matches)

        for match in matches:
            yield self._build_entity(segment, match)

    @staticmethod
    def _filter_overlapping_matches(matches: list[_Match]) -> list[_Match]:
        """Remove overlapping matches by keeping matches with best score then max
        length among overlapping matches
        """
        matches.sort(key=lambda m: (m.score, m.length), reverse=True)
        matches_filtered = []
        for match in matches:
            if any(match.overlaps(prev_match) for prev_match in matches_filtered):
                continue
            matches_filtered.append(match)

        # restore initial order
        matches_filtered.sort(key=lambda m: m.start)
        return matches_filtered

    def _build_entity(self, segment: Segment, match: _Match) -> Entity:
        """Build an entity from a match in a segment"""
        # extract text and spans corresponding to match
        text, spans = span_utils.extract(segment.text, segment.spans, [(match.start, match.end)])

        # create entity
        label = match.rule.label
        entity = Entity(label=label, text=text, spans=spans)
        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])

        # propagate attrs_to_copy from segment to new entity
        for attr_label in self.attrs_to_copy:
            for attr in segment.attrs.get(label=attr_label):
                copied_attr = attr.copy()
                entity.attrs.add(copied_attr)
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(copied_attr, self.description, [attr])

        # create normalization attributes for each
        # normalization descriptor of the rule
        for norm in match.rule.normalizations:
            norm_attr = norm.to_attribute(match.score)
            entity.attrs.add(norm_attr)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(norm_attr, self.description, source_data_items=[segment])

        return entity


def build_simstring_matcher_databases(
    simstring_db_file: Path,
    rules_db_file: Path,
    rules: Iterable[BaseSimstringMatcherRule],
):
    """Generate the databases needed by :class:`BaseSimstringMatcher`.

    Parameters
    ----------
    simstring_db_file : Path
        Database used by the fuzzy matching `simstring` library.
    rules_db_file : Path
        `shelve` database storing the mapping between terms to match and
        corresponding BaseSimstringMatcherRule` objects (one term to match may
        correspond to several rules)
    rules : iterable of BaseSimstringMatcherRule
        Rules to add to databases
    """
    # the params passed to simstring.writer are copy/pasted from QuickUMLS
    # cf https://github.com/Georgetown-IR-Lab/QuickUMLS/blob/a3ba0b3559da2574a907f4d41aa0f2c1c0d5ce0a/quickumls/toolbox.py#L173
    simstring_db_writer = simstring.writer(
        str(simstring_db_file),
        3,  # unit of character n-grams
        False,  # represent begin and end of strings in n-grams
        True,  # use unicode mode
    )

    # writeback=True needed because we are updating the values in the mapping,
    # not just writing
    rules_db = shelve.open(str(rules_db_file), flag="n", writeback=True)  # noqa: S301

    # add rules to databases
    for rule in rules:
        term_to_match = rule.term

        # apply preprocessing
        term_to_match = anyascii(term_to_match.lower())

        # add to simstring db
        simstring_db_writer.insert(term_to_match)
        # add to rules db
        if term_to_match not in rules_db:
            rules_db[term_to_match] = []
        rules_db[term_to_match].append(rule)
    simstring_db_writer.close()
    rules_db.sync()
    rules_db.close()


_TOKENIZATION_PATTERN = re.compile(r"[\w]+|[^\w ]")


def _build_candidate_ranges_with_regexp(text: str, min_length: int, max_length: int) -> Iterator[tuple[int, int]]:
    """From a string, generate all candidate matches (by tokenizing it and then
    re-concatenating tokens) and return their ranges. Based on the QuickUMLS
    code.

    Parameters
    ----------
    text : str
        Text from which to generate candidates
    min_length : int
        Min length of a candidate, in characters
    max_length : int
        Max length of a candidate, in characters

    Returns
    -------
    iterator of tuple
        Iterator over ranges of candidate matches

    Examples
    --------
    >>> text = "I have type 2 diabetes"
    >>> ranges = _build_candidate_ranges_with_regexp(text, 2, 10)
    >>> candidates = [text[slice(*r)] for r in ranges]
    >>> candidates
    ['I have', 'have', 'have type', 'type', 'type 2', '2 diabetes', 'diabetes']
    """
    # find all tokens and corresponding ranges using regexp
    tokens_and_ranges = [(m.group(0), m.span()) for m in _TOKENIZATION_PATTERN.finditer(text)]
    if len(tokens_and_ranges) == 0:
        return
    tokens, ranges = zip(*tokens_and_ranges)
    nb_tokens = len(tokens)

    # iterate over non-empty tokens
    for i in range(nb_tokens):
        start_token = tokens[i]
        if not start_token[0].isalnum():
            continue
        # build candidate by appending next tokens
        start = ranges[i][0]
        for j in range(i, nb_tokens):
            end_token = tokens[j]
            # next token is empty, skip candidate
            if not end_token[0].isalnum():
                continue
            end = ranges[j][1]
            length = end - start
            # candidate is too short, skip
            if length < min_length:
                continue
            # candidate is too long, stop appending tokens
            if length > max_length:
                break
            yield (start, end)


def _build_candidate_ranges_with_spacy(
    spacy_doc: Any,
    min_length: int,
    max_length: int,
) -> Iterator[tuple[int, int]]:
    """From a pre-tokenized spacy Document, generate all candidate matches (by
    concatenating tokens) and return their ranges, filtering out some tokens
    based on their part-of-speech tags. Based on the QuickUMLS code.

    This will often give better results than _build_candidate_ranges_with_regexp()
    because the part-of-speech filtering allows us to avoid some false positives.

    Parameters
    ----------
    spacy_doc : Any
        Spacy document of text from which to generate candidates
    min_length : int
        Min length of a candidate, in characters
    max_length : int
        Max length of a candidate, in characters

    Returns
    -------
    iterator of tuple
        Iterator over ranges of candidate matches

    Examples
    --------
    >>> text = "I have type 2 diabetes"
    >>> doc = spacy.blank("en")(text)
    >>> ranges = _build_candidate_ranges_with_spacy(doc, 2, 10)
    >>> candidates = [text[slice(*r)] for r in ranges]
    >>> candidates
    ['I have', 'have', 'have type', 'type', 'type 2', '2 diabetes', 'diabetes']
    """

    # don't allow candidates to start or end with pre/post positions,
    # determinants or conjunctions
    def is_invalid_boundary_token(token):
        return token.is_punct or token.is_space or token.pos_ in ("ADP", "DET", "SCONJ", "CCONJ", "CONJ")

    # iterate over tokens
    nb_tokens = len(spacy_doc)
    for i in range(nb_tokens):
        start_token = spacy_doc[i]
        if is_invalid_boundary_token(start_token):
            continue
        # build candidate by appending next tokens
        for j in range(i, nb_tokens):
            # don't allow candidates made of only one word that is a stop word
            if i == j and start_token.is_stop:
                continue
            end_token = spacy_doc[j]
            if is_invalid_boundary_token(end_token):
                continue
            span = spacy_doc[i : j + 1]
            length = span.end_char - span.start_char
            # candidate is too short, skip
            if length < min_length:
                continue
            # candidate is too long, stop appending tokens
            if length > max_length:
                break
            yield (span.start_char, span.end_char)


# based on https://github.com/Georgetown-IR-Lab/QuickUMLS/blob/master/quickumls/toolbox.py


def _get_similarity_score(text_1, text_2, similarity_name, ngram_size=3):
    """The MIT License (MIT)

    Copyright (c) 2019 Georgetown Information Retrieval Lab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def _make_ngrams(text, ngram_size):
        n = len(text) if len(text) < ngram_size else ngram_size
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    ngrams_1 = _make_ngrams(text_1, ngram_size)
    ngrams_2 = _make_ngrams(text_2, ngram_size)
    nb_ngrams_1 = len(ngrams_1)
    nb_ngrams_2 = len(ngrams_2)
    nb_ngrams_common = len(ngrams_1.intersection(ngrams_2))

    if similarity_name == "dice":
        return 2 * nb_ngrams_common / (nb_ngrams_1 + nb_ngrams_2)
    if similarity_name == "jaccard":
        return nb_ngrams_common / (nb_ngrams_1 + nb_ngrams_2 - nb_ngrams_common)
    if similarity_name == "cosine":
        return nb_ngrams_common / math.sqrt(nb_ngrams_1 * nb_ngrams_2)
    assert similarity_name == "overlap", 'similarity_name should be one of ["dice", "jaccard", "cosine", "overlap"]'
    return nb_ngrams_common
