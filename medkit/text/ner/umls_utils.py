from __future__ import annotations

__all__ = [
    "UMLSEntry",
    "load_umls_entries",
    "preprocess_term_to_match",
    "preprocess_acronym",
    "guess_umls_version",
    "SEMGROUPS",
    "SEMGROUP_LABELS",
]

import dataclasses
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from anyascii import anyascii
from tqdm import tqdm

SEMGROUP_LABELS = {
    "ACTI": "activity",
    "ANAT": "anatomy",
    "CHEM": "chemical",
    "CONC": "concept",
    "DEVI": "device",
    "DISO": "disorder",
    "GENE": "genes_sequence",
    "GEOG": "geographic_area",
    "LIVB": "living_being",
    "OBJC": "object",
    "OCCU": "occupation",
    "ORGA": "organization",
    "PHEN": "phenomenon",
    "PHYS": "physiology",
    "PROC": "procedure",
}
"""
Labels corresponding to UMLS semgroups
"""


SEMGROUPS = list(SEMGROUP_LABELS.keys())
"""
Valid UMLS semgroups
"""


@dataclasses.dataclass
class UMLSEntry:
    """Entry in MRCONSO.RRF file of a UMLS dictionary

    Attributes
    ----------
    cui : str
        Unique identifier of the concept designated by the term
    term : str
        Original version of the term
    semtypes : list of str, optional
        Semantic types of the concept (TUIs)
    semgroups : list of str, optional
        Semantic groups of the concept
    """

    cui: str
    term: str
    semtypes: list[str] | None = None
    semgroups: list[str] | None = None

    def to_dict(self):
        return {
            "cui": self.cui,
            "term": self.term,
            "semtypes": self.semtypes,
            "semgroups": self.semgroups,
        }


# based on https://github.com/GanjinZero/CODER/blob/master/coderpp/test/load_umls.py


def load_umls_entries(
    mrconso_file: str | Path,
    mrsty_file: str | Path | None = None,
    sources: list[str] | None = None,
    languages: list[str] | None = None,
    show_progress: bool = False,
) -> Iterator[UMLSEntry]:
    """Load all terms and associated CUIs found in a UMLS MRCONSO.RRF file

    Parameters
    ----------
    mrconso_file : str or Path
        Path to the UMLS MRCONSO.RRF file
    mrsty_file : str or Path, optional
        Path to the UMLS MRSTY.RRF file. If provided, semtypes info will be
        included in the entries returned.
    sources : list of str, optional
        Sources to consider (ex: ICD10, CCS) If none provided, CUIs and terms
        of all sources will be taken into account.
    languages : list of str, optional
        Languages to consider. If none provided, CUIs and terms of all languages
        will be taken into account
    show_progress : bool, default=False
        Whether to show a progressbar

    Returns
    -------
    iterator of UMLSEntry
        Iterator over all term entries found in UMLS install
    """
    mrconso_file = Path(mrconso_file)
    if mrsty_file is not None:
        mrsty_file = Path(mrsty_file)

    file_size = mrconso_file.stat().st_size
    luis_seen = set()

    # load semtypes and semgroups if MRSTY is provided
    if mrsty_file is not None:
        semtypes_by_cui = load_semtypes_by_cui(mrsty_file)
        semgroups_by_semtype = load_semgroups_by_semtype()
    else:
        semtypes_by_cui = None
        semgroups_by_semtype = None

    with mrconso_file.open(encoding="utf-8") as fp:
        lines_iter = fp

        if show_progress:
            progress_bar = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )

        for line in lines_iter:
            if show_progress:
                line_size = len(line.encode("utf-8"))
                progress_bar.update(line_size)

            row = line.strip().split("|")
            cui = row[0]
            language = row[1]
            lui = row[3]
            source = row[11]
            term = row[14]

            if sources is not None and source not in sources:
                continue
            if languages is not None and language not in languages:
                continue
            if lui in luis_seen:
                continue

            if semtypes_by_cui is not None and cui in semtypes_by_cui:
                semtypes = semtypes_by_cui[cui]
                semgroups = [semgroups_by_semtype[semtype] for semtype in semtypes]
            else:
                semtypes = None
                semgroups = None

            luis_seen.add(lui)
            yield UMLSEntry(cui, term, semtypes, semgroups)

    if show_progress:
        progress_bar.close()


def load_semtypes_by_cui(mrsty_file: str | Path) -> dict[str, list[str]]:
    """Load the list of semtypes associated to each CUI found in a MRSTY.RRF file

    Params
    ------
    mrsty_file : str or Path
        Path to the UMLS MRSTY.RRF file.

    Returns
    -------
    dict of str to list of str
        Mapping between CUIs and associated semtypes
    """
    semtypes_by_cui = defaultdict(list)

    with Path(mrsty_file).open() as fp:
        for line in fp:
            row = line.strip().split("|")
            cui = row[0]
            semtypes_by_cui[cui].append(row[1])

    return dict(semtypes_by_cui)


# The semantic groups provide a partition of the UMLS Metathesaurus for 99.5%
# of the concepts, we use this file to obtain a semtype-to-semgroup mapping.
# Source: UMLS project
# https://lhncbc.nlm.nih.gov/semanticnetwork/download/sg_archive/SemGroups-v04.txt
_UMLS_SEMGROUPS_FILE = Path(__file__).parent / "umls_semgroups_v04.txt"
_SEMGROUPS_BY_SEMTYPE = None


def load_semgroups_by_semtype() -> dict[str, str]:
    """Load the semgroup associated to each semtype

    Returns
    -------
    Dict[str, str]
        Mapping between semtype TUIs and corresponding semgroup
    """
    global _SEMGROUPS_BY_SEMTYPE  # noqa: PLW0603
    if _SEMGROUPS_BY_SEMTYPE is None:
        _SEMGROUPS_BY_SEMTYPE = {}
        with Path(_UMLS_SEMGROUPS_FILE).open() as fp:
            for line in fp:
                semgroup, _, semtype, _ = line.split("|")
                _SEMGROUPS_BY_SEMTYPE[semtype] = semgroup
    return _SEMGROUPS_BY_SEMTYPE


_BRACKET_PATTERN = re.compile("\\(.*?\\)")


def preprocess_term_to_match(
    term: str,
    lowercase: bool,
    normalize_unicode: bool,
    clean_nos: bool = True,
    clean_brackets: bool = False,
    clean_dashes: bool = False,
):
    """Preprocess a UMLS term for matching purposes

    Parameters
    ----------
    term: str
        Term to preprocess
    lowercase : bool
        Whether `term` should be lowercased
    normalize_unicode : bool
        Whether `term_to_match` should be ASCII-only (non-ASCII chars replaced by closest ASCII chars)
    clean_nos : bool, default=True
        Whether to remove "NOS"
    clean_brackets : bool, default=False
        Whether to remove brackets
    clean_dashes : bool, default=False
        Whether to remove dashes
    """
    if lowercase:
        term = term.lower()
    if normalize_unicode:
        term = anyascii(term)

    term = " " + term + " "
    if clean_nos:
        term = term.replace(" NOS ", " ").replace(" nos ", " ")
    if clean_brackets:
        term = _BRACKET_PATTERN.sub("", term)
    if clean_dashes:
        term = term.replace("-", " ")
    return " ".join([w for w in term.split() if w])


_ACRONYM_PATTERN = re.compile(r"^ *(?P<acronym>[^ \(\)]+) *\( *(?P<expanded>[^\(\)]+) *\) *$")


def preprocess_acronym(term: str) -> str | None:
    """Detect if a term contains an acronym with the expanded form between
    parenthesis, and return the acronym if that is the case.

    This will work for terms such as: "ECG (ÉlectroCardioGramme)", where the
    acronym can be rebuilt by taking the ASCII version of each uppercase
    letter inside the parenthesis.

    Parameters
    ----------
    term : str
        Term that may contain an acronym. Ex: "ECG (ÉlectroCardioGramme)"

    Returns
    -------
    str, optional
        The acronym in the term if any, else `None`. Ex: "ECG"
    """
    match = _ACRONYM_PATTERN.match(term)
    if not match:
        return None

    # extract acronym (before the parenthesis) and expanded form (between parenthesis)
    acronym = match.group("acronym")
    expanded = match.group("expanded")

    # try to rebuild acronym from expanded form:
    # replace special characters with ASCII
    expanded = anyascii(expanded)
    # keep only uppercase chars
    acronym_candidate = "".join(c for c in expanded if c.isupper())
    # if it doesn't match the part before the parenthesis
    # we decide it is not an acronym
    if acronym != acronym_candidate:
        return None
    return acronym


def guess_umls_version(path: str | Path) -> str:
    """Try to infer UMLS version (ex: "2021AB") from any UMLS-related path

    Parameters
    ----------
    path : str or Path
        Path to the root directory of the UMLS install or any file inside that directory

    Returns
    -------
    str
        UMLS version, estimated by finding the leaf-most folder in `path` that is not
        "META", "NET" nor "LEX", nor a subfolder of these folders
    """
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    while any(dir_name in path.parts for dir_name in ("META", "NET", "LEX")):
        path = path.parent
    return path.name
