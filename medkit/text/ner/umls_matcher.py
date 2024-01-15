from __future__ import annotations

__all__ = ["UMLSMatcher"]

import dataclasses
import logging
from pathlib import Path
from typing import Iterator, Sequence

import yaml
from typing_extensions import Literal

from medkit.text.ner import umls_utils
from medkit.text.ner._base_simstring_matcher import (
    BaseSimstringMatcher,
    BaseSimstringMatcherNormalization,
    BaseSimstringMatcherRule,
    build_simstring_matcher_databases,
)

_CACHE_PARAMS_FILENAME = "params.yml"
_RULES_DB_FILENAME = "rules"
_SIMSTRING_DB_FILENAME = "simstring"

_SPACY_LANGUAGE_MAP = {
    "ENG": "en",
    "GER": "de",
    "SPA": "es",
    "POR": "pt",
    "FRE": "fr",
    "ITA": "it",
    "DUT": "nl",
}

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _UMLSMatcherCacheParams:
    umls_version: str
    language: str
    semgroups: list[str]
    labels_by_semgroup: dict[str]
    lowercase: bool
    normalize_unicode: bool


class UMLSMatcher(BaseSimstringMatcher):
    """Entity annotator identifying UMLS concepts using the `simstring` fuzzy
    matching algorithm (http://chokkan.org/software/simstring/).

    This operation is heavily inspired by the `QuickUMLS` library
    (https://github.com/Georgetown-IR-Lab/QuickUMLS).

    By default, only terms belonging to the `ANAT` (anatomy), `CHEM` (Chemicals &
    Drugs), `DEVI` (Devices), `DISO` (Disorders), `PHYS` (Physiology) and `PROC`
    (Procedures) semgroups will be considered. This behavior can be changed with
    the `semgroups` parameter.

    Note that setting `spacy_tokenization_language` to `True` might reduce the
    number of false positives. This requires the `spacy` optional dependency,
    which can be installed with `pip install medkit-lib[spacy]`.
    """

    _SEMGROUP_BY_SEMTYPE = None

    def __init__(
        self,
        umls_dir: str | Path,
        cache_dir: str | Path,
        language: str,
        threshold: float = 0.9,
        min_length: int = 3,
        max_length: int = 50,
        similarity: Literal["cosine", "dice", "jaccard", "overlap"] = "jaccard",
        lowercase: bool = True,
        normalize_unicode: bool = False,
        spacy_tokenization: bool = False,
        semgroups: Sequence[str] = (
            "ANAT",
            "CHEM",
            "DEVI",
            "DISO",
            "PHYS",
            "PROC",
        ),
        blacklist: list[str] | None = None,
        same_beginning: bool = False,
        output_labels_by_semgroup: str | dict[str, str] | None = None,
        attrs_to_copy: list[str] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        umls_dir : str or Path
            Path to the UMLS directory containing the MRCONSO.RRF and
            MRSTY.RRF files.
        cache_dir : str or Path
            Path to the directory into which the umls database will be cached.
            If it doesn't exist yet, the database will be automatically
            generated (it can take a long time) and stored there, ready to be
            reused on further instantiations. If it already exists, a check will
            be done to make sure the params used when the database was generated
            are consistent with the params of the current instance. If you want
            to rebuild the database with new params using the same cache dir,
            you will have to manually delete it first.
        language : str
            Language to consider as found in the MRCONSO.RRF file. Example:
            `"FRE"`. Will trigger a regeneration of the database if changed.
        threshold : float, default=0.9
            Minimum similarity threshold (between 0.0 and 1.0) between a UMLS term
            and the text of a matched entity.
        min_length : int, default=3
            Minimum number of chars in matched entities.
        max_length : int, default=50
            Maximum number of chars in matched entities.
        similarity : str, default="jaccard"
            Similarity metric to use.
        lowercase : bool, default=True
            Whether to use lowercased versions of UMLS terms and input entities
            (except for acronyms for which the uppercase term is always used).
            Will trigger a regeneration of the database if changed.
        normalize_unicode : bool, default=False
            Whether to use ASCII-only versions of UMLS terms and input entities
            (non-ASCII chars replaced by closest ASCII chars). Will trigger a
            regeneration of the database if changed.
        spacy_tokenization : bool, default=False
            If `True`, spacy will be used to tokenize input segments and filter
            out some tokens based on their part-of-speech tags, such as
            determinants, conjunctions and prepositions. If `None`, a simple
            regexp based tokenization will be used, which is faster but might
            give more false positives.
        semgroups : sequence of str, default=("ANAT", "CHEM", "DEVI", "DISO", "PHYS", "PROC")
            Ids of UMLS semantic groups that matched concepts should belong to.
            :see: https://lhncbc.nlm.nih.gov/semanticnetwork/download/sg_archive/SemGroups-v04.txt
            If set to `None`, all concepts can be matched.
            Will trigger a regeneration of the database if changed.
        blacklist : list of str, optional
            Optional list of exact terms to ignore.
        same_beginning : bool, default=False
            Ignore all matches that start with a different character than the
            term of the rule. This can be convenient to get rid of false
            positives on words that are very similar but have opposite meanings
            because of a preposition, for instance "activation" and
            "inactivation".
        output_labels_by_semgroup : str or dict, optional
            By default, ~`medkit.text.ner.umls.SEMGROUP_LABELS` will be used as
            entity labels. Use this parameter to override them. Example:
            `{"DISO": "problem", "PROC": "test}`. If `output_labels_by_semgroup`
            is a string, all entities will use this string as label instead.
            Will trigger a regeneration of the database if changed.
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the source
            segment to the created entity. Useful for propagating context
            attributes (negation, antecedent, etc)
        name : str, optional
            Name describing the matcher (defaults to the class name).
        uid : str, optional
            Identifier of the matcher.
        """
        umls_dir = Path(umls_dir)
        cache_dir = Path(cache_dir)

        # check that values of semgroups are valid semgroup ids
        if semgroups is not None:
            for semgroup in semgroups:
                if semgroup not in umls_utils.SEMGROUPS:
                    msg = f"Unknown semgroup: {semgroup}. Should be one of {umls_utils.SEMGROUPS}"
                    raise ValueError(msg)

        cache_dir.mkdir(parents=True, exist_ok=True)

        labels_by_semgroup = self._get_labels_by_semgroup(output_labels_by_semgroup)

        cache_params = _UMLSMatcherCacheParams(
            umls_version=umls_utils.guess_umls_version(umls_dir),
            language=language,
            semgroups=list(semgroups),
            labels_by_semgroup=labels_by_semgroup,
            lowercase=lowercase,
            normalize_unicode=normalize_unicode,
        )

        cache_params_file = cache_dir / _CACHE_PARAMS_FILENAME
        simstring_db_file = cache_dir / _SIMSTRING_DB_FILENAME
        rules_db_file = cache_dir / _RULES_DB_FILENAME

        if cache_params_file.exists():
            with cache_params_file.open() as fp:
                existing_cache_params = _UMLSMatcherCacheParams(**yaml.safe_load(fp))
            if cache_params != existing_cache_params:
                msg = (
                    f"Cache directory {cache_dir} contains database pre-computed"
                    f" with different params: {existing_cache_params} vs"
                    f" {cache_params}"
                )
                raise ValueError(msg)
        else:
            logger.info("Building simstring database from UMLS terms, this may take a while")
            rules = self._build_rules(
                umls_dir,
                language,
                lowercase,
                normalize_unicode,
                set(semgroups),
                labels_by_semgroup,
            )

            build_simstring_matcher_databases(simstring_db_file, rules_db_file, rules)

            with cache_params_file.open(mode="w") as fp:
                yaml.safe_dump(dataclasses.asdict(cache_params), fp)

        if spacy_tokenization:
            spacy_tokenization_language = _SPACY_LANGUAGE_MAP.get(language)
            if spacy_tokenization_language is None:
                msg = "Spacy tokenization not supported for language" f" '{spacy_tokenization_language}'"
                raise ValueError(msg)
        else:
            spacy_tokenization_language = None

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

    @classmethod
    def _get_labels_by_semgroup(cls, output_labels: str | dict[str, str] | None) -> dict[str, str]:
        """Return a mapping giving the label to use for all entries of a given semgroup

        output_labels : str or dict of str to str, optional
            Optional mapping of labels to use. Can be used to override the default
            labels. If `output_labels` is a single string, it will be used as a unique
            label for all semgroups

        Returns
        -------
        dict of str to str
            A mapping with semgroups as keys and corresponding label as values
        """
        if output_labels is None:
            return umls_utils.SEMGROUP_LABELS

        if isinstance(output_labels, str):
            return {key: output_labels for key in umls_utils.SEMGROUP_LABELS}

        # check that the keys of output_labels are valid semgroup ids
        for semgroup in output_labels:
            if semgroup not in umls_utils.SEMGROUPS:
                msg = f"Unknown semgroup: {semgroup}. Should be one of {umls_utils.SEMGROUPS}"
                raise ValueError(msg)

        label_mapping = umls_utils.SEMGROUP_LABELS.copy()
        label_mapping.update(output_labels)
        return label_mapping

    @classmethod
    def _build_rules(
        cls,
        umls_dir: Path,
        language: str,
        lowercase: bool,
        normalize_unicode: bool,
        semgroups: set[str] | None,
        labels_by_semgroup: dict[str, str],
    ) -> Iterator[BaseSimstringMatcherRule]:
        """Create rules for all UMLS entries (filtered by `language` and
        `semgroups`) with appropriate labels (based on `labels_by_semgroup`)
        """
        # get iterator to all UMLS entries
        entries_iter = umls_utils.load_umls_entries(
            mrconso_file=umls_dir / "MRCONSO.RRF",
            mrsty_file=umls_dir / "MRSTY.RRF",
            languages=[language],
            show_progress=True,
        )

        version = umls_utils.guess_umls_version(umls_dir)

        for entry in entries_iter:
            # filter out entries not belonging to allowed semgroups
            entry_semgroups = entry.semgroups
            if semgroups is not None:
                entry_semgroups = [s for s in entry_semgroups if s in semgroups]
                if len(entry_semgroups) == 0:
                    continue

            # take label corresponding to semgroup (1st semgroup if multiple)
            semgroup = entry_semgroups[0]
            label = labels_by_semgroup[semgroup]

            norm = BaseSimstringMatcherNormalization(
                kb_name="umls", kb_version=version, kb_id=entry.cui, term=entry.term
            )

            # acronym detection
            acronym = umls_utils.preprocess_acronym(entry.term)
            if acronym is not None:
                term = acronym
            else:
                # perform UMLS-specific cleaning (lowercase and normalize unicode
                # will be handled by BaseSimstringMatcher)
                term = umls_utils.preprocess_term_to_match(
                    entry.term,
                    lowercase=False,
                    normalize_unicode=False,
                )

            # keep case sensitivity for terms that are all uppercase to avoid
            # too many false positives
            case_sensitive = True if term.isupper() else not lowercase

            rule = BaseSimstringMatcherRule(
                term=term,
                label=label,
                case_sensitive=case_sensitive,
                unicode_sensitive=not normalize_unicode,
                normalizations=[norm],
            )
            yield rule
