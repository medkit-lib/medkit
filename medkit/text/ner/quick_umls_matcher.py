"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[quick-umls-matcher]`.
"""
from __future__ import annotations

__all__ = ["QuickUMLSMatcher"]

from typing import TYPE_CHECKING, ClassVar, Iterator, NamedTuple

import quickumls.about
import quickumls.constants
from packaging.version import parse as parse_version
from quickumls import QuickUMLS
from typing_extensions import Literal

from medkit.core.text import (
    Entity,
    NEROperation,
    Segment,
    UMLSNormAttribute,
    span_utils,
)
from medkit.text.ner import umls_utils

if TYPE_CHECKING:
    from pathlib import Path

# workaround for https://github.com/Georgetown-IR-Lab/QuickUMLS/issues/68
_spacy_language_map_fixed = False


def _fix_spacy_language_map():
    global _spacy_language_map_fixed  # noqa: PLW0603
    if _spacy_language_map_fixed:
        return

    if parse_version(quickumls.about.__version__) < parse_version("1.4.1"):
        for key, value in quickumls.constants.SPACY_LANGUAGE_MAP.items():
            ext = "_core_web_sm" if value == "en" else "_core_news_sm"
            quickumls.constants.SPACY_LANGUAGE_MAP[key] = value + ext

    _spacy_language_map_fixed = True


class _QuickUMLSInstall(NamedTuple):
    version: str
    language: str
    lowercase: bool
    normalize_unicode: bool


class QuickUMLSMatcher(NEROperation):
    """Entity annotator relying on QuickUMLS.

    This annotator requires a QuickUMLS installation performed
    with `python -m quickumls.install` with flags corresponding
    to the params `language`, `version`, `lowercase` and `normalize_unicode`
    passed at init. QuickUMLS installations must be registered with the
    `add_install` class method.

    For instance, if we want to use `QuickUMLSMatcher` with a french
    lowercase QuickUMLS install based on UMLS version 2021AB,
    we must first create this installation with:

    >>> python -m quickumls.install --language FRE --lowercase /path/to/umls/2021AB/data /path/to/quick/umls/install

    then register this install with:

    >>> QuickUMLSMatcher.add_install(
    >>>        "/path/to/quick/umls/install",
    >>>        version="2021AB",
    >>>        language="FRE",
    >>>        lowercase=True,
    >>> )

    and finally instantiate the matcher with:

    >>> matcher = QuickUMLSMatcher(
    >>>     version="2021AB",
    >>>     language="FRE",
    >>>     lowercase=True,
    >>> )

    This mechanism makes it possible to store in the OperationDescription
    how the used QuickUMLS was created, and to reinstantiate the same matcher
    on a different environment if a similar install is available.
    """

    _install_paths: ClassVar[dict[_QuickUMLSInstall, str]] = {}

    @classmethod
    def add_install(
        cls,
        path: str | Path,
        version: str,
        language: str,
        lowercase: bool = False,
        normalize_unicode: bool = False,
    ):
        """Register path and settings of a QuickUMLS installation performed
        with `python -m quickumls.install`

        Parameters
        ----------
        path : str or Path
            The path to the destination folder passed to the install command
        version : str
            The version of the UMLS database, for instance "2021AB"
        language : str
            The language flag passed to the install command, for instance "ENG"
        lowercase : bool, default=False
            Whether the --lowercase flag was passed to the install command
            (concepts are lowercased to increase recall)
        normalize_unicode : bool, default=False
            Whether the --normalize-unicode flag was passed to the install command
            (non-ASCII chars in concepts are converted to the closest ASCII chars)
        """
        install = _QuickUMLSInstall(version, language, lowercase, normalize_unicode)
        cls._install_paths[install] = str(path)

    @classmethod
    def clear_installs(cls):
        """Remove all QuickUMLS installation registered with `add_install`"""
        cls._install_paths.clear()

    @classmethod
    def _get_path_to_install(
        cls,
        version: str,
        language: str,
        lowercase: bool = False,
        normalize_unicode: bool = False,
    ) -> str:
        """Find a QuickUMLS install with corresponding settings

        The QuickUMLS install must have been previously registered with `add_install`.
        """
        install = _QuickUMLSInstall(version, language, lowercase, normalize_unicode)
        path = cls._install_paths.get(install)
        if not path:
            msg = (
                f"Couldn't find any Quick- UMLS install for version={version},"
                f" language={language}, lowercase={lowercase},"
                f" normalize_unicode={normalize_unicode}.\nRegistered installs:"
                f" {cls._install_paths}"
            )
            raise ValueError(msg)
        return path

    def __init__(
        self,
        version: str,
        language: str,
        lowercase: bool = False,
        normalize_unicode: bool = False,
        overlapping: Literal["length", "score"] = "length",
        threshold: float = 0.9,
        window: int = 5,
        similarity: Literal["dice", "jaccard", "cosine", "overlap"] = "jaccard",
        accepted_semtypes: list[str] = quickumls.constants.ACCEPTED_SEMTYPES,
        attrs_to_copy: list[str] | None = None,
        output_label: str | dict[str, str] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Instantiate the QuickUMLS matcher

        Parameters
        ----------
        version : str
            UMLS version of the QuickUMLS install to use, for instance "2021AB"
            Will be used to decide with QuickUMLS to use
        language : str
            Language flag of the QuickUMLS install to use, for instance "ENG".
            Will be used to decide with QuickUMLS to use
        lowercase : bool, default=False
            Whether to use a QuickUMLS install with lowercased concepts
            Will be used to decide with QuickUMLS to use
        normalize_unicode : bool, default=False
            Whether to use a QuickUMLS install with non-ASCII chars concepts
            converted to the closest ASCII chars.
            Will be used to decide with QuickUMLS to use
        overlapping : {"length", "score"}, default="length"
            Criteria for sorting multiple potential matches (cf QuickUMLS doc)
        threshold : float, default=0.9
            Minimum similarity (cf QuickUMLS doc)
        window : int, default=5
            Max number of tokens per match (cf QuickUMLS doc)
        similarity : {"dice", "jaccard", "cosine", "overlap"}, default="jaccard"
            Similarity measure to use (cf QuickUMLS doc)
        accepted_semtypes : list of str, optional
            UMLS semantic types that matched concepts should belong to (cf QuickUMLS doc).
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the source segment
            to the created entity. Useful for propagating context attributes
            (negation, antecendent, etc)
        output_label : str or dict of str to str, optional
            By default, ~`medkit.text.ner.umls.SEMGROUP_LABELS` will be used as
            entity labels. Use this parameter to override them. Example:
            `{"DISO": "problem", "PROC": "test}`. If `output_labels_by_semgroup`
            is a string, all entities will use this string as label instead.
        name : str, optional
            Name describing the matcher (defaults to the class name)
        uid : str, optional
            Identifier of the matcher
        """
        _fix_spacy_language_map()

        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.language = language
        self.version = version
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.overlapping = overlapping
        self.threshold = threshold
        self.similarity = similarity
        self.window = window
        self.accepted_semtypes = accepted_semtypes
        self.attrs_to_copy = attrs_to_copy

        path_to_install = self._get_path_to_install(version, language, lowercase, normalize_unicode)
        self._matcher = QuickUMLS(
            quickumls_fp=path_to_install,
            overlapping_criteria=overlapping,
            threshold=threshold,
            window=window,
            similarity_name=similarity,
            accepted_semtypes=accepted_semtypes,
        )
        assert (  # noqa: PT018
            self._matcher.language_flag == language
            and self._matcher.to_lowercase_flag == lowercase
            and self._matcher.normalize_unicode_flag == normalize_unicode
        ), "Inconsistent QuickUMLS install flags"

        self._semtype_to_semgroup = umls_utils.load_semgroups_by_semtype()
        self.label_mapping = self._get_label_mapping(output_label)

    @staticmethod
    def _get_label_mapping(output_label: None | str | dict[str, str]) -> dict[str, str]:
        """Return label mapping according to `output_label`"""
        if output_label is None:
            return umls_utils.SEMGROUP_LABELS

        if isinstance(output_label, str):
            return {key: output_label for key in umls_utils.SEMGROUP_LABELS}

        if isinstance(output_label, dict):
            label_mapping = umls_utils.SEMGROUP_LABELS.copy()
            label_mapping.update(output_label)
            return label_mapping
        return None

    def run(self, segments: list[Segment]) -> list[Entity]:
        """Return entities (with UMLS normalization attributes) for each match in `segments`

        Parameters
        ----------
        segments : list of Segment
            List of segments into which to look for matches

        Returns
        -------
        list of Entity
            Entities found in `segments`, with :class:`~UMLSNormAttribute` attributes.
        """
        return [entity for segment in segments for entity in self._find_matches_in_segment(segment)]

    def _find_matches_in_segment(self, segment: Segment) -> Iterator[Entity]:
        matches = self._matcher.match(segment.text)
        for match_candidates in matches:
            # only the best matching CUI (1st match candidate) is returned
            # TODO should we create a normalization attributes for each CUI instead?
            match = match_candidates[0]

            text, spans = span_utils.extract(segment.text, segment.spans, [(match["start"], match["end"])])
            semtypes = list(match["semtypes"])

            # define label using the first semtype
            semgroup = self._semtype_to_semgroup[semtypes[0]]
            label = self.label_mapping[semgroup]

            entity = Entity(
                label=label,
                text=text,
                spans=spans,
            )

            for attr_label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=attr_label):
                    copied_attr = attr.copy()
                    entity.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            norm_attr = UMLSNormAttribute(
                cui=match["cui"],
                umls_version=self.version,
                term=match["term"],
                score=match["similarity"],
                sem_types=list(match["semtypes"]),
            )
            entity.attrs.add(norm_attr)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])
                self._prov_tracer.add_prov(norm_attr, self.description, source_data_items=[segment])

            yield entity
