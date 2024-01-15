"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[metrics-diarization]`.
"""

__all__ = ["DiarizationEvaluator", "DiarizationEvaluatorResult"]

import dataclasses
import logging
from typing import Sequence

# When pyannote and spacy are both installed, a conflict might occur between the
# ujson library used by pandas (a pyannote dependency) and the ujson library used
# by srsrly (a spacy dependency), especially in docker environments.
# srsly seems to end up using the ujson library from pandas, which is older and does not
# support the escape_forward_slashes parameters, instead of its own.
# The bug seems to only happen when pandas is imported from pyannote, not if
# we import pandas manually first.
# So as a workaround, we always import pandas before importing something from pyannote
import pandas as pd  # noqa: F401
from pyannote.core.annotation import Annotation as PAAnnotation
from pyannote.core.annotation import Segment as PASegment
from pyannote.core.annotation import Timeline as PATimeline
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

from medkit.core.audio import AudioDocument, Segment

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DiarizationEvaluatorResult:
    """Results returned by :class:`~.DiarizationEvaluator`

    Attributes
    ----------
    der : float
        Diarization Error Rate, combination of confusion, false alarm and missed
        detection
    confusion : float
        Ratio of time detected as speech but attributed to a wrong speaker
        (over `total_speech`)
    false_alarm : float
        Ratio of time corresponding to non-speech mistakenly detected as
        speech (over `total_speech`)
    missed_detection : float
        Ratio of time corresponding to undetected speech (over `total_speech`)
    total_speech : float
        Total duration of speech in the reference
    support : float
        Total duration of audio
    """

    der: float
    confusion: float
    false_alarm: float
    missed_detection: float
    total_speech: float
    support: float


class DiarizationEvaluator:
    """Diarization Error Rate (DER) computation based on `pyannote`.

    The DER is the ratio of time that is not attributed correctly because of
    one of the following errors:

    - detected as non-speech when there was speech (missed detection);

    - detected as speech where there was none (false alarm);

    - attributed to the wrong speaker (confusion).

    This component expects as input reference documents containing the reference
    speech turn segments as well as corresponding predicted speech turn
    segments. The speech turn segments must each have a speaker attribute.

    Note that values of the reference and predicted speaker attributes (ie
    speaker labels) don't have to be the same, since they will be optimally
    remapped using the Hungarian algorithm.
    """

    def __init__(
        self,
        turn_label: str = "turn",
        speaker_label: str = "speaker",
        collar: float = 0.0,
    ):
        """Parameters
        ----------
        turn_label : str, default="turn"
            Label of the turn segments on the reference documents
        speaker_label : str, default="speaker"
            Label of the speaker attributes on the reference and predicted turn segments
        collar : float, default=0.0
            Margin of error (in seconds) around start and end of reference segments
        """
        self.turn_label = turn_label
        self.speaker_label = speaker_label
        self.collar = collar

    def compute(
        self,
        reference: Sequence[AudioDocument],
        predicted: Sequence[Sequence[Segment]],
    ) -> DiarizationEvaluatorResult:
        """Compute and return the DER for predicted speech turn segments, against
        reference annotated documents.

        Parameters
        ----------
        reference : sequence of AudioDocument
            Reference documents containing speech turns segments with
            `turn_label` as label, each of them containing a speaker attribute
            with `speaker_label` as label.
        predicted : sequence of sequence of Segment
            Predicted segments containing each a speaker attribute with
            `speaker_label` as label. This is a list of list that must be of the
            same length and ordering as `reference`.

        Returns
        -------
        DiarizationEvaluatorResult
            Computed metrics
        """
        if len(reference) != len(predicted):
            msg = "Reference and predicted must have the same length"
            raise ValueError(msg)

        # init pyannote metrics object into which results are accumulated
        pa_metric = GreedyDiarizationErrorRate(collar=self.collar)
        support = 0.0

        for ref_doc, pred_segs in zip(reference, predicted):
            support += ref_doc.audio.duration
            ref_segs = ref_doc.anns.get(label=self.turn_label)
            # UEM timeline representing annotated timeline
            # (needed to get rid of pyannote warning)
            uem = PATimeline(segments=[PASegment(start=0.0, end=ref_doc.audio.duration)])

            # convert reference and predicted segment to pyannote annotation objects
            ref_pa_ann = self._get_pa_annotation(ref_segs)
            pred_pa_ann = self._get_pa_annotation(pred_segs)
            # pass them to pyannote metrics object
            pa_metric(
                reference=ref_pa_ann,
                hypothesis=pred_pa_ann,
                uem=uem,
            )

        # retrieve accumulated results from pyannote metrics object,
        # in fractional form
        return DiarizationEvaluatorResult(
            der=abs(pa_metric),
            confusion=pa_metric["confusion"] / pa_metric["total"],
            false_alarm=pa_metric["false alarm"] / pa_metric["total"],
            missed_detection=pa_metric["missed detection"] / pa_metric["total"],
            total_speech=pa_metric["total"],
            support=support,
        )

    def _get_pa_annotation(self, segments: Sequence[Segment]) -> PAAnnotation:
        """Convert list of medkit speech turn segments with speaker attribute to
        pyannote annotation object
        """
        pa_ann = PAAnnotation()

        for i, seg in enumerate(segments):
            # retrieve speaker
            speaker_attrs = seg.attrs.get(label=self.speaker_label)

            if not speaker_attrs:
                msg = f"Attribute with label '{self.speaker_label}' not found on turn segment"
                raise ValueError(msg)
            if len(speaker_attrs) > 1:
                logger.warning(
                    "Found several attributes with label '%s' ignoring all but first",
                    self.speaker_label,
                )
            speaker = speaker_attrs[0].value

            # create pyannote segment object to hold boundaries
            # and add it to pyannote annotation object
            pa_seg = PASegment(seg.span.start, seg.span.end)
            pa_ann[pa_seg, i] = speaker

        return pa_ann
