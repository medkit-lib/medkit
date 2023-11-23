__all__ = ["PowerNormalizer"]

from typing import List, Optional

import numpy as np

from medkit.core.audio import PreprocessingOperation, Segment, MemoryAudioBuffer

_EPS = 1e-12  # epsilon value to avoid zero-div


class PowerNormalizer(PreprocessingOperation):
    """Normalization operation setting the RMS power of each audio signal to a target value."""

    def __init__(
        self,
        output_label: str,
        target_value: float = 1.0,
        channel_wise: bool = False,
        uid: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        output_label:
            Label of output normalized segments.
        target_value:
            Value to set the RMS power of each segment to.
        channel_wise:
            If `True`, the normalization is performed per-channel, thus modifying
            the balance of multichannel signals.
        uid:
            Identifier of the normalizer.
        """

        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        self.channel_wise = channel_wise
        self.target_value = target_value

    def run(self, segments: List[Segment]) -> List[Segment]:
        """Return a normalized segment for each segment in `segments`.

        Parameters
        ----------
        segments:
            Audio segments to normalize.

        Returns
        -------
        List[~medkit.core.audio.Segment]:
            Power-normalized segments, one per segment in `segments`.
        """
        return [self._normalize_segment(s) for s in segments]

    def _normalize_segment(self, segment: Segment) -> Segment:
        audio = segment.audio
        signal = audio.read(copy=True)
        if self.channel_wise:
            std = np.std(signal, axis=1).reshape((audio.nb_channels, -1))
            signal /= (std + _EPS) / self.target_value
        else:
            signal /= (np.std(signal) + _EPS) / self.target_value

        normalized_audio = MemoryAudioBuffer(signal, sample_rate=audio.sample_rate)
        normalized_segment = Segment(
            label=self.output_label,
            span=segment.span,
            audio=normalized_audio,
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(normalized_segment, self.description, [segment])

        return normalized_segment
