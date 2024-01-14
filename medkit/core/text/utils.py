from __future__ import annotations

__all__ = [
    "replace_point_after_keywords",
    "replace_multiple_newline_after_sentence",
    "replace_newline_inside_sentence",
    "clean_newline_character",
    "clean_multiple_whitespaces_in_sentence",
    "clean_parentheses_eds",
    "replace_point_in_uppercase",
    "replace_point_in_numbers",
    "replace_point_before_keywords",
    "lstrip",
    "rstrip",
    "strip",
]


import re
from typing import TYPE_CHECKING

from medkit.core.text import span_utils

if TYPE_CHECKING:
    from medkit.core.text.span import AnySpan

# Some strings for character classification
_NUMERIC_CHARS = "0-9"
_UPPERCASE_CHARS = "A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸÆŒ"
_PUNCT_CHARS = r"\.,;\?\!\:\("
_LOWERCASE_CHARS = "a-zàâäçéèêëîïôöùûüÿ"


def clean_newline_character(text: str, spans: list[AnySpan], keep_endlines: bool = False) -> tuple[str, list[AnySpan]]:
    """Replace the newline character depending on its position in the text.
    The endlines characters that are not suppressed can be either kept as
    endlines, or replaced by spaces. This method combines :func:`replace_multiple_newline_after_sentence`
    and :func:`replace_newline_inside_sentence`.

    Parameters
    ----------
    text : str
        The text to be modified
    spans : list of AnySpan
        Spans associated to the `text`
    keep_endlines : bool, default=False
        Whether to keep the endlines as '.\\\\n' or replace them with '. '

    Returns
    -------
    text : str
        The cleaned text
    spans : list of AnySpan
        The list of modified spans

    Examples
    --------
    >>> text = "This is\\n\\n\\ta sentence\\nAnother\\nsentence\\n\\nhere"
    >>> spans = [Span(0, len(text))]
    >>> text, spans = clean_newline_character(text, spans, keep_endlines=False)
    >>> print(text)
    This is a sentence. Another sentence here

    >>> text, spans = clean_newline_character(text, spans, keep_endlines=True)
    >>> print(text)
    This is a sentence.
    Another sentence here
    """
    text, spans = replace_multiple_newline_after_sentence(text, spans)
    text, spans = replace_newline_inside_sentence(text, spans)
    text, spans = _replace_text(text, spans, pattern="\n+", repl=".\n" if keep_endlines else ". ")
    return text, spans


def clean_parentheses_eds(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Modify the text near the parentheses depending on its content.
    The rules are adapted for French documents.

    Examples
    --------
    >>> text = \"\"\"
    ... Le test PCR est (-), pas de nouvelles.
    ... L'examen d'aujourd'hui est (+).
    ... Les bilans réalisés (biologique, métabolique en particulier à la recherche
    ... de GAMT et X fragile) sont revenus négatifs.
    ... Le patient a un traitement(debuté le 3/02).
    ... \"\"\"
    >>> spans = [Span(0, len(text))]
    >>> text, spans = clean_parentheses_eds(text, spans)
    >>> print(text)
    Le test PCR est  negatif , pas de nouvelles.
    L'examen d'aujourd'hui est  positif .
    Les bilans réalisés sont revenus négatifs ; biologique, métabolique en particulier à la recherche
    de GAMT et X fragile.
    Le patient a un traitement,debuté le 3/02,.
    """
    text, spans = _replace_text(text, spans, r"\(-\)", " negatif ", group=0)
    text, spans = _replace_text(text, spans, r"\(\+\)", " positif ", group=0)

    text, spans = _replace_big_parentheses(text, spans)
    text, spans = _replace_small_parentheses(text, spans)
    return text, spans


def clean_multiple_whitespaces_in_sentence(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Replace multiple white-spaces between alphanumeric characters and
    lowercase characters with a single whitespace

    Example:
    -------
    >>> text = "A   phrase    with  multiple   spaces     "
    >>> spans = [Span(0, len(text))]
    >>> text, spans = clean_multiple_whitespaces_in_sentence(text, spans)
    >>> print(text)
    A phrase with multiple spaces
    """
    pattern = r"([ \t]{2,})"
    text, spans = _replace_text(text, spans, pattern, " ", group=0)
    return text, spans


def replace_point_after_keywords(
    text: str,
    spans: list[AnySpan],
    keywords: list[str],
    strict: bool = False,
    replace_by: str = " ",
) -> tuple[str, list[AnySpan]]:
    """Replace the character '.' after a keyword and update its span.
    Could be used to replace dots that indicate the title of a person (i.e. M. or Mrs.)
    or some dots that appear by mistake after `keywords`

    Parameters
    ----------
    text : str
        The text to be modified
    spans : list of AnySpan
        Spans associated to the `text`
    keywords : list of str
        Word or pattern to match before a point
    strict : bool, default=False
        If True, the keyword must be followed by a point.
        If False, the keyword could have zero or many whitespaces before a point
    replace_by : str, default=" "
        Replacement string

    Returns
    -------
    text : str
        The text with the replaced matches
    spans : list of AnySpan
        The list of modified spans

    Examples
    --------
    >>> text = "Le Dr. a un rdv. Mme. Bernand est venue à 14h"
    >>> spans = [Span(0, len(text))]
    >>> keywords = ["Dr", "Mme"]
    >>> text, spans = replace_point_after_keywords(text, spans, keywords, replace_by="")
    >>> print(text)
    Le Dr a un rdv. Mme Bernand est venue à 14h
    """
    # Create a list regex using '\b' to indicate that keyword is a word
    keywords_regexp = "|".join([rf"\b{keyword}" for keyword in keywords])
    pattern = rf"(?:{keywords_regexp})(\.)" if strict else rf"(?:{keywords_regexp})(\s*\.)"

    # The first group has the span of interest
    text, spans = _replace_text(text, spans, pattern, repl=replace_by, group=1)
    return text, spans


def replace_multiple_newline_after_sentence(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Replace multiple space characters between a newline
    character \\\\n and a capital letter or a number with a single newline character.

    Parameters
    ----------
    text : str
        The text to be modified
    spans : list of AnySpan
        Spans associated to the `text`

    Returns
    -------
    text : str
        The cleaned text
    spans : list of AnySpan
        The list of modified spans
    """
    pattern = rf"(?P<blanks>\r?\n[\r\n]*)[\t\s]*[{_NUMERIC_CHARS}{_UPPERCASE_CHARS}]"
    replace_by = "\n"
    text, spans = _replace_text(text, spans, pattern, repl=replace_by, group="blanks")
    return text, spans


def replace_newline_inside_sentence(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Replace the newline character \\\\n between lowercase letters
    or punctuation marks with a space

    Parameters
    ----------
    text : str
        The text to be modified
    spans : list of AnySpan
        Spans associated to the `text`

    Returns
    -------
    text : str
        The cleaned text
    spans : list of AnySpan
        The list of modified spans
    """
    pattern = rf"(?P<blanks>\r?\n[\r\n]*)[\t\s]*[{_LOWERCASE_CHARS}{_PUNCT_CHARS}]"
    replace_by = " "
    text, spans = _replace_text(text, spans, pattern, repl=replace_by, group="blanks")
    return text, spans


def _replace_big_parentheses(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Modify the sentence containing large parentheses.
    The new sentence contains the text after the parentheses followed by
    the text that was inside the parentheses.
    """
    # capture multiple spaces to control the output format
    pattern = re.compile(r"(\s*)\((?P<txt_inside>[^)(]{30,5000})\)(\s*)(?P<txt_after>[^.]*)\.")

    while True:
        # iteration over the new text until no matches are found
        match = pattern.search(text)
        if match is None:
            break

        # extract groups including their spans
        txt_in, span_in = span_utils.extract(text, spans, [match.span("txt_inside")])
        txt_af, span_af = span_utils.extract(text, spans, [match.span("txt_after")])

        if span_af:
            # insert characters before and after each group
            txt_in, span_in = span_utils.insert(txt_in, span_in, [len(txt_in)], ["."])
            # insert a space by default (eq: ' {text_af} ; ')
            txt_af, span_af = span_utils.insert(txt_af, span_af, [0, len(txt_af)], [" ", " ; "])
            # create the new phrase
            txt_new, span_new = span_utils.concatenate([txt_af, txt_in], [span_af, span_in])
        else:
            # there is no text after (), insert ';' before
            txt_new, span_new = span_utils.insert(txt_in, span_in, [0, len(txt_in)], [" ; ", "."])

        # add the new phrase into the text. Extract text_before and text_after
        # from this match and concatenate all to update texp_tmp and spans
        txt_before, span_before = span_utils.extract(text, spans, [(0, match.start(0))])
        txt_after, span_after = span_utils.extract(text, spans, [(match.end(0), len(text))])
        text, spans = span_utils.concatenate([txt_before, txt_new, txt_after], [span_before, span_new, span_after])
    return text, spans


def _replace_small_parentheses(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Modify the sentence containing small parentheses.
    The new sentence has the text that was inside the parentheses surrounded by `,`
    """
    pattern = r"(\()(?:[^)(]{1,29})(\))"
    # capture each parenthesis
    group_1 = [match.span(1) for match in re.finditer(pattern, text)]
    group_2 = [match.span(2) for match in re.finditer(pattern, text)]
    ranges = sorted([*group_1, *group_2], key=lambda sp: sp[0])
    text, spans = span_utils.replace(text, spans, ranges, [","] * len(ranges))
    return text, spans


def _replace_text(
    text: str,
    spans: list[AnySpan],
    pattern: str,
    repl: str,
    group: str | int = 0,
) -> tuple[str, list[AnySpan]]:
    """Replace matches in `text` by `repl` and update its spans."""
    ranges = [(match.span(group)) for match in re.finditer(pattern, text)]
    return span_utils.replace(text, spans, ranges, [repl] * len(ranges))


def replace_point_in_uppercase(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Replace the character '.' between uppercase characters
    with a space and update its span.

    Examples
    --------
    >>> text = "Abréviation ING.DRT or RTT.J"
    >>> spans = [Span(0, len(text))]
    >>> text, spans = replace_point_in_uppercase(text, spans)
    >>> print(text)
    Abréviation ING DRT or RTT J

    """
    pattern = rf"[{_UPPERCASE_CHARS}](\.)[{_UPPERCASE_CHARS}]"
    text, spans = _replace_text(text, spans, pattern, " ", group=1)
    return text, spans


def replace_point_in_numbers(text: str, spans: list[AnySpan]) -> tuple[str, list[AnySpan]]:
    """Replace the character '.' between numbers
    with the character ',' a space and update its span.

    Example:
    -------
    >>> text = "La valeur est de 3.456."
    >>> spans = [Span(0, len(text))]
    >>> text, spans = replace_point_in_numbers(text, spans)
    >>> print(text)
    La valeur est de 3,456.
    """
    pattern = rf"[{_NUMERIC_CHARS}](\.)[{_NUMERIC_CHARS}]"
    text, spans = _replace_text(text, spans, pattern, ",", group=1)
    return text, spans


def replace_point_before_keywords(text: str, spans: list[AnySpan], keywords: list[str]) -> tuple[str, list[AnySpan]]:
    """Replace the character '.' before a keyword
    with a space and update its span.
    """
    keywords_regexp = "|".join([rf"{keyword}\b" for keyword in keywords])
    pattern = rf"(\s\.\s*)(?:{keywords_regexp})"
    text, spans = _replace_text(text, spans, pattern, " ", group=1)
    return text, spans


def lstrip(text: str, start: int = 0, chars: str | None = None) -> tuple[str, int]:
    """Returns a copy of the string with leading characters removed
    and its corresponding new start index.

    Parameters
    ----------
    text : str
        The text to strip.
    start : int, default=0
        The start index from the original text if any.
    chars : str, optional
        The list of characters to strip. Default behaviour is like `str.lstrip([chars])`.

    Returns
    -------
    new_text : str
        New text
    new_start : int
        New start index
    """
    new_text = text.lstrip(chars)
    new_start = start + (len(text) - len(new_text))
    return new_text, new_start


def rstrip(text: str, end: int | None = None, chars: str | None = None) -> tuple[str, int]:
    """Returns a copy of the string with trailing characters removed
    and its corresponding new end index.

    Parameters
    ----------
    text : str
        The text to strip.
    end : int, optional
        The end index from the original text if any.
    chars : str, optional
        The list of characters to strip. Default behaviour is like `str.rstrip([chars])`.

    Returns
    -------
    new_text : str
        New text
    new_end : int
        New end index
    """
    if end is None:
        end = len(text)
    new_text = text.rstrip(chars)
    new_end = end - (len(text) - len(new_text))
    return new_text, new_end


def strip(text: str, start: int = 0, chars: str | None = None) -> tuple[str, int, int]:
    """Returns a copy of the string with leading characters removed
    and its corresponding new start and end indexes.

    Parameters
    ----------
    text : str
        The text to strip.
    start : int, default=0
        The start index from the original text if any.
    chars : str, optional
        The list of characters to strip. Default behaviour is like `str.lstrip([chars])`.

    Returns
    -------
    new_text : str
        New text
    new_start : int
        New start index
    new_end : int
        New end index
    """
    new_text, new_start = lstrip(text, start, chars)
    new_end = new_start + len(new_text)
    new_text, new_end = rstrip(new_text, new_end)
    return new_text, new_start, new_end
