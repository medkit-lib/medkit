from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Optional

from medkit._import import import_optional
from medkit.core.text import Entity, span_utils
from medkit.core.text.operation import NEROperation

if TYPE_CHECKING:
    from medkit.core.text import Segment

presidio_analyzer = import_optional("presidio_analyzer", extra="deid")

from presidio_analyzer import Pattern, PatternRecognizer

__all__ = ["PIIDetector", "FrDateRecognizer","FrAgeRecognizer"]



class PIIDetector(NEROperation):
    """Classify sensitive text information."""

    def __init__(self, uid: str | None = None, name: str | None = None, **kwargs):
        super().__init__(uid=uid, name=name, **kwargs)
        self._analyzer = presidio_analyzer.AnalyzerEngine()

    def run(self, segments: list[Segment]) -> list[Entity]:
        return [entity for segment in segments for entity in self._run_one(segment)]

    def _run_one(self, segment: Segment) -> Iterator[Entity]:
        for result in self._analyzer.analyze(text=segment.text, language="en"):
            text, spans = span_utils.extract(
                text=segment.text, spans=segment.spans, ranges=[(result.start, result.end)]
            )
            yield Entity(
                label=result.entity_type,
                text=text,
                spans=spans,
                metadata={"score": result.score},
            )

    
class FrDateRecognizer(PatternRecognizer):
    """
    Recognizes French Date using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """
    PATTERNS = [
        Pattern(
            "French dates with day month year",
            r"\b(?i)(\d{1,2}|1er) ((janvier|janv.|JAN)|(février|févr.|FÉV)|(mars|MAR)|(avril|avr.|AVR)|(mai|MAI)|(juin|JUN)|(juillet|juill.|JUL)|(août|AOÛ)|(septembre|sept.|SEP)|(octobre|oct.|OCT)|(novembre|nov.|NOV)|(décembre|déc|DÉC)) \d{4}\b",
            0.9
        ),
        Pattern(
            "French dates with month year",
            r"\b(?i)((janvier|janv.|JAN)|(février|févr.|FÉV)|(mars|MAR)|(avril|avr.|AVR)|(mai|MAI)|(juin|JUN)|(juillet|juill.|JUL)|(août|AOÛ)|(septembre|sept.|SEP)|(octobre|oct.|OCT)|(novembre|nov.|NOV)|(décembre|déc|DÉC)) \d{4}\b",
            0.9
        ),
        Pattern(
            "French dates with day month",
            r"\b(?i)(\d{1,2} |1er )((janvier|janv.|JAN)|(février|févr.|FÉV)|(mars|MAR)|(avril|avr.|AVR)|(mai|MAI)|(juin|JUN)|(juillet|juill.|JUL)|(août|AOÛ)|(septembre|sept.|SEP)|(octobre|oct.|OCT)|(novembre|nov.|NOV)|(décembre|déc|DÉC))\b",
            0.9
        ),
        Pattern(
            "French dates with month",
            r"\b(?i)((janvier|janv.|JAN)|(février|févr.|FÉV)|(mars|MAR)|(avril|avr.|AVR)|(mai|MAI)|(juin|JUN)|(juillet|juill.|JUL)|(août|AOÛ)|(septembre|sept.|SEP)|(octobre|oct.|OCT)|(novembre|nov.|NOV)|(décembre|déc|DÉC))\b",
            0.9
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        supported_language: str = "fr",
        supported_entity : str = "FR_DATE",
    ):
        patterns = patterns if patterns else self.PATTERNS
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            supported_language=supported_language,
        )

class FrAgeRecognizer(PatternRecognizer):
    """
    Recognizes French Phone Age using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """
    PATTERNS = [
        Pattern(
            "French age written out numbers .. (ans|mois)' ou ' .. et .. ans' ou ' .. à .. ans' ou ' .. ou .. ans'",
             r"\d+\s+(ans|mois)|\d+\s+et\s+\d+\s+(ans|mois)|\d+\s+à\s+\d+\s+(ans|mois)|\d+\s+ou\s+\d+\s+(ans|mois)",
            0.9
        ),
        Pattern(
            "French age written out numbers 'ans et .. mois' ou 'ans et .. ou .. mois' ou 'ans et .. à .. mois' ",
            r"\d+\s+ans\s+et\s+\d+\s+mois|\d+\s+ans\s+et\s+\d+\s+à\s+\d+\s+mois|\d+\s+ans\s+et\s+\d+\s+ou\s+\d+\s+mois",
            0.9
        ),
        Pattern(
            "French age written out with letters '.. (ans|mois)' ou ' .. et .. ans' ou ' .. à .. ans' ou ' .. ou .. ans' ",
            r"((dix|vingt|trente|quarante|cinquante|soixante|soixante-dix|quatre-vingt|quatre-vingt-dix|cent)(\s+|-|-et-)?)?(un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize)?\s+((et\s+|à\s+|ou\s+)\d+\s+)?(ans|mois)",
            0.9
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        supported_language: str = "fr",
        supported_entity : str = "FR_AGE",
    ):
        patterns = patterns if patterns else self.PATTERNS
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            supported_language=supported_language,
        )