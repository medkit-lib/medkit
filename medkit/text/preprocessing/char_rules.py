__all__ = [
    "ALL_CHAR_RULES",
    "LIGATURE_RULES",
    "FRACTION_RULES",
    "SIGN_RULES",
    "SPACE_RULES",
    "DOT_RULES",
    "QUOTATION_RULES",
]

#: Rules for ligatures
LIGATURE_RULES = [
    ("\u00c6", "AE"),
    ("\u00e6", "ae"),
    ("\u0152", "OE"),
    ("\u0153", "oe"),
]
#: Rules for fraction characters
FRACTION_RULES = [
    ("\u00bc", "1/4"),
    ("\u00bd", "1/2"),
    ("\u00be", "3/4"),
    ("\u2150", "1/7"),
    ("\u2151", "1/9"),
    ("\u2152", "1/10"),
    ("\u2153", "1/3"),
    ("\u2154", "2/3"),
    ("\u2155", "1/5"),
    ("\u2156", "2/5"),
    ("\u2157", "3/5"),
    ("\u2158", "4/5"),
    ("\u2159", "1/6"),
    ("\u215a", "5/6"),
    ("\u215b", "1/8"),
    ("\u215c", "3/8"),
    ("\u215d", "5/8"),
    ("\u215e", "7/8"),
    ("\u2189", "0/3"),
]
#: Rules for non-standard spaces
SPACE_RULES = [
    ("\u00a0", " "),
    ("\u1680", " "),
    ("\u2002", " "),
    ("\u2003", " "),
    ("\u2004", " "),
    ("\u2005", " "),
    ("\u2006", " "),
    ("\u2007", " "),
    ("\u2008", " "),
    ("\u2009", " "),
    ("\u200a", " "),
    ("\u200b", " "),
    ("\u202f", " "),
    ("\u205f", " "),
    ("\u2420", " "),
    ("\u3000", " "),
    ("\u303f", " "),
    ("\ufeff", " "),
]

#: Rules for sign chars
SIGN_RULES = [
    ("\u00a9", ""),  # copyright
    ("\u00ae", ""),  # registered
    ("\u2122", ""),  # trade
]

#: Rules for dot chars
DOT_RULES = [
    # horizontal ellipsis
    ("\u2026", "..."),
    ("\u22ef", "..."),
]

#: RegexpReplacer quotation marks: replace double and single quotation marks
QUOTATION_RULES = [
    ("»", '"'),  # normalize double quotation marks
    ("«", '"'),  # replace double quotation marks
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\u201e", '"'),
    ("\u201f", '"'),
    ("\u2039", '"'),
    ("\u203a", '"'),
    ("\u02f5", '"'),
    ("\u02f6", '"'),
    ("\u02dd", '"'),
    ("\uff02", '"'),
    ("\u201a", ""),  # single low quotation (remove)
    ("\u2018", "'"),  # left side single quotation
    ("\u2019", "'"),  # right side single quotation
    ("\u201b", "'"),  # single high reverse quotation
    ("\u02ca", "'"),  # grave accent
    ("\u0060", "'"),
    ("\u02cb", "'"),  # acute accent
    ("\u00b4", "'"),
]

#: All pre-defined rules for CharReplacer
ALL_CHAR_RULES = DOT_RULES + FRACTION_RULES + LIGATURE_RULES + QUOTATION_RULES + SIGN_RULES + SPACE_RULES
