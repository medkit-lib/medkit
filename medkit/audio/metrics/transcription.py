"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[metrics-transcription]`.
"""
from __future__ import annotations

__all__ = ["TranscriptionEvaluator", "TranscriptionEvaluatorResult"]

import dataclasses
import functools
import logging
import string
from typing import TYPE_CHECKING, Sequence

from speechbrain.utils.metric_stats import ErrorRateStats

from medkit.text.utils.decoding import get_ascii_from_unicode

if TYPE_CHECKING:
    from medkit.core.audio import AudioDocument, Segment

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TranscriptionEvaluatorResult:
    """Results returned by :class:`~.TranscriptionEvaluator`

    Attributes
    ----------
    wer : float
        Word Error Rate, combination of word insertions, deletions and
        substitutions
    word_insertions : float
        Ratio of extra words in prediction (over `word_support`)
    word_deletions : float
        Ratio of missing words in prediction (over `word_support`)
    word_substitutions : float
        Ratio of replaced words in prediction (over `word_support`)
    word_support : int
        Total number of words
    cer : float
        Character Error Rate, same as `wer` but at character level
    char_insertions : float
        Identical to `word_insertions` but at character level
    char_deletions : float
        Identical to `word_deletions` but at character level
    char_substitutions : float
        Identical to `word_substitutions` but at character level
    char_support : int
        Total number of characters (not including whitespaces, post punctuation
        removal and unicode replacement)
    """

    wer: float
    word_insertions: float
    word_deletions: float
    word_substitutions: float
    word_support: int
    cer: float
    char_insertions: float
    char_deletions: float
    char_substitutions: float
    char_support: int


class TranscriptionEvaluator:
    """Word Error Rate (WER) and Character Error Rate (CER) computation based on
    `speechbrain`.

    The WER is the ratio of predictions errors at the word level, taking into
    accounts:

    - words present in the reference transcription but missing from the
      prediction;

    - extra predicted words not present in the reference;

    - reference words mistakenly replaced by other words in the prediction.

    The CER is identical to the WER but computed at the character level rather
    than at the word level.

    This component expects as input reference documents containing speech
    segments with reference transcription attributes, as well as corresponding
    speech segments with predicted transcription attributes.
    """

    def __init__(
        self,
        speech_label: str = "speech",
        transcription_label: str = "transcription",
        case_sensitive: bool = False,
        remove_punctuation: bool = True,
        replace_unicode: bool = False,
    ):
        """Parameters
        ----------
        speech_label : str, default="speech"
            Label of the speech segments on the reference documents
        transcription_label : str, default="transcription"
            Label of the transcription attributes on the reference and predicted
            speech segments
        case_sensitive : bool, default=False
            Whether to take case into consideration when comparing reference and
            prediction
        remove_punctuation : bool, default=True
            If True, punctuation in reference and predictions is removed before
            comparing (based on `string.punctuation`)
        replace_unicode : bool, default=False
            If True, special unicode characters in reference and predictions are
            replaced by their closest ASCII characters (when possible) before
            comparing
        """
        self.speech_label = speech_label
        self.transcription_label = transcription_label
        self.case_sensitive = case_sensitive
        self.remove_punctuation = remove_punctuation
        self.replace_unicode = replace_unicode

    def compute(
        self,
        reference: Sequence[AudioDocument],
        predicted: Sequence[Sequence[Segment]],
    ) -> TranscriptionEvaluatorResult:
        """Compute and return the WER and CER for predicted transcription
        attributes, against reference annotated documents.

        Parameters
        ----------
        reference : sequence of AudioDocument
            Reference documents containing speech segments with `speech_label`
            as label, each of them containing a transcription attribute with
            `transcription_label` as label.
        predicted : sequence of sequence of Segment
            Predicted segments containing each a transcription attribute with
            `transcription_label` as label. This is a list of list that must be
            of the same length and ordering as `reference`.

        Returns
        -------
        TranscriptionEvaluatorResult
            Computed metrics
        """
        if len(reference) != len(predicted):
            msg = "Reference and predicted must have the same length"
            raise ValueError(msg)

        sb_wer_metric = ErrorRateStats()
        sb_cer_metric = ErrorRateStats(split_tokens=True)

        for i, (ref_doc, pred_segs) in enumerate(zip(reference, predicted)):
            ref_segs = ref_doc.anns.get(label=self.speech_label)
            ref_words = self._convert_speech_segs_to_words(ref_segs)
            pred_words = self._convert_speech_segs_to_words(pred_segs)

            sb_wer_metric.append(ids=[i], predict=[pred_words], target=[ref_words])
            sb_cer_metric.append(ids=[i], predict=[pred_words], target=[ref_words])

        wer_results = sb_wer_metric.summarize()
        nb_words = wer_results["num_scored_tokens"]
        cer_results = sb_cer_metric.summarize()
        nb_chars = cer_results["num_scored_tokens"]

        return TranscriptionEvaluatorResult(
            wer=wer_results["num_edits"] / nb_words,
            word_insertions=wer_results["insertions"] / nb_words,
            word_deletions=wer_results["deletions"] / nb_words,
            word_substitutions=wer_results["substitutions"] / nb_words,
            word_support=nb_words,
            cer=cer_results["num_edits"] / nb_chars,
            char_insertions=cer_results["insertions"] / nb_chars,
            char_deletions=cer_results["deletions"] / nb_chars,
            char_substitutions=cer_results["substitutions"] / nb_chars,
            char_support=nb_chars,
        )

    def _convert_speech_segs_to_words(self, segments: Sequence[Segment]) -> list[str]:
        """Convert list of speech segments with transcription attribute to list of
        words that can be passed to speechbrain metrics objects
        """
        # get values of all transcription attributes and concatenate them into
        # one big string representing the transcription of the whole document

        # sort segments by time to concatenate in correct order
        segments = sorted(segments, key=lambda s: s.span)
        texts = []
        for seg in segments:
            # retrieve transcription
            transcription_attrs = seg.attrs.get(label=self.transcription_label)

            if not transcription_attrs:
                msg = f"Attribute with label '{self.transcription_label}' not found on speech segment"
                raise ValueError(msg)
            if len(transcription_attrs) > 1:
                logger.warning(
                    "Found several attributes with label '%s' ignoring all but first", self.transcription_label
                )
            transcription = transcription_attrs[0].value
            texts.append(transcription)

        text = " ".join(texts)

        # apply pre-WER transforms
        if not self.case_sensitive:
            text = text.lower()
        if self.remove_punctuation:
            punct_trans_table = _get_punctation_translation_table()
            text = text.translate(punct_trans_table)
        if self.replace_unicode:
            text = get_ascii_from_unicode(text, logger=logger)

        # split into words
        return [w for w in text.split(" ") if w]


@functools.lru_cache
def _get_punctation_translation_table():
    """Return a translation table mapping all punctuations chars to a single space,
    that can be used with `str.translate()`
    """
    return str.maketrans(dict.fromkeys(string.punctuation, " "))
