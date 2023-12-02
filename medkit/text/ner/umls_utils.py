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
from typing import Dict, Iterator, List, Optional, Union

import unidecode
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
    cui:
        Unique identifier of the concept designated by the term
    ref_term:
        Original version of the term
    semtypes:
        Semantic types of the concept (TUIs)
    semgroups:
        Semantic groups of the concept
    """

    cui: str
    term: str
    semtypes: Optional[List[str]] = None
    semgroups: Optional[List[str]] = None

    def to_dict(self):
        return dict(
            cui=self.cui,
            term=self.term,
            semtypes=self.semtypes,
            semgroups=self.semgroups,
        )


# based on https://github.com/GanjinZero/CODER/blob/master/coderpp/test/load_umls.py


def load_umls_entries(
    mrconso_file: Union[str, Path],
    mrsty_file: Union[str, Path] = None,
    sources: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    show_progress: bool = False,
) -> Iterator[UMLSEntry]:
    """Load all terms and associated CUIs found in a UMLS MRCONSO.RRF file

    Parameters
    ----------
    mrconso_file:
        Path to the UMLS MRCONSO.RRF file
    mrsty_file:
        Path to the UMLS MRSTY.RRF file. If provided, semtypes info will be
        included in the entries returned.
    sources:
        Sources to consider (ex: ICD10, CCS) If none provided, CUIs and terms
        of all sources will be taken into account.
    languages:
        Languages to consider. If none provided, CUIs and terms of all languages
        will be taken into account
    show_progress:
        Whether to show a progressbar

    Returns
    -------
    Iterator[UMLSEntry]
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

    with open(mrconso_file, encoding="utf-8") as fp:
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


def load_semtypes_by_cui(mrsty_file: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Load the list of semtypes associated to each CUI found in a MRSTY.RRF file

    Params
    ------
    mrsty_file:
        Path to the UMLS MRSTY.RRF file.

    Returns
    -------
    Dict[str, List[str]]
        Mapping between CUIs and associated semtypes
    """

    mrsty_file = Path(mrsty_file)
    semtypes_by_cui = defaultdict(list)

    with open(mrsty_file) as fp:
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


def load_semgroups_by_semtype() -> Dict[str, str]:
    """
    Load the semgroup associated to each semtype

    Returns
    -------
    Dict[str, str]
        Mapping between semtype TUIs and corresponding semgroup
    """

    global _SEMGROUPS_BY_SEMTYPE
    if _SEMGROUPS_BY_SEMTYPE is None:
        _SEMGROUPS_BY_SEMTYPE = {}
        with open(_UMLS_SEMGROUPS_FILE) as fp:
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
    """
    Preprocess a UMLS term for matching purposes

    Parameters
    ----------
    term: str
        Term to preprocess
    lowercase:
        Whether `term` should be lowercased
    normalize_unicode:
        Whether `term_to_match` should be ASCII-only (non-ASCII chars replaced by closest ASCII chars)
    clean_nos:
        Whether to remove "NOS"
    clean_brackets:
        Whether to remove brackets
    clean_dashes:
        Whether to remove dashes
    """
    if lowercase:
        term = term.lower()
    if normalize_unicode:
        term = unidecode.unidecode(term)

    term = " " + term + " "
    if clean_nos:
        term = term.replace(" NOS ", " ").replace(" nos ", " ")
    if clean_brackets:
        term = _BRACKET_PATTERN.sub("", term)
    if clean_dashes:
        term = term.replace("-", " ")
    term = " ".join([w for w in term.split() if w])
    return term


_ACRONYM_PATTERN = re.compile(
    r"^ *(?P<acronym>[^ \(\)]+) *\( *(?P<expanded>[^\(\)]+) *\) *$"
)


def preprocess_acronym(term: str) -> Optional[str]:
    """
    Detect if a term contains an acronym with the expanded form between
    parenthesis, and return the acronym if that is the case.

    This will work for terms such as: "ECG (ÉlectroCardioGramme)", where the
    acronym can be rebuilt by taking the ASCII version of each uppercase
    letter inside the parenthesis.

    Parameters
    ----------
    term:
        Term that may contain an acronym. Ex: "ECG (ÉlectroCardioGramme)"

    Returns
    -------
    Optional[str]
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
    expanded = unidecode.unidecode(expanded)
    # keep only uppercase chars
    acronym_candidate = "".join(c for c in expanded if c.isupper())
    # if it doesn't match the part before the parenthesis
    # we decide it is not an acronym
    if acronym != acronym_candidate:
        return None
    return acronym


def guess_umls_version(path: Union[str, Path]) -> str:
    """Try to infer UMLS version (ex: "2021AB") from any UMLS-related path

    Parameters
    ----------
    path:
        Path to the root directory of the UMLS install or any file inside that directory
    Returns
    -------
        UMLS version, estimated by finding the leaf-most folder in `path` that is not
        "META", "NET" nor "LEX", nor a subfolder of these folders
    """
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    while any(dir_name in path.parts for dir_name in ("META", "NET", "LEX")):
        path = path.parent
    return path.name
