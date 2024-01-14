"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[resampler]`.
"""
from __future__ import annotations

__all__ = ["Resampler"]

import resampy

from medkit.core.audio import MemoryAudioBuffer, PreprocessingOperation, Segment


class Resampler(PreprocessingOperation):
    """Resampling operation relying on the resampy package."""

    def __init__(
        self,
        output_label: str,
        sample_rate: int,
        fast: bool = False,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        output_label : str
            Label of output resampled segments.
        sample_rate : int
            Target sample rate to resample to, in samples per second.
        fast : bool, default=False
            If `True`, prefer speed over quality and use resampy's "kaiser_fast" filter
            instead of "kaiser_best".
        uid : str, optional
            Identifier of the resampler.
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        self.sample_rate = sample_rate
        self.fast = fast

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Return a resampled segment for each segment in `segments`.

        Parameters
        ----------
        segments : list of Segment
            Audio segments to resample.

        Returns
        -------
        list of Segment
            Resampled segments, one per segment in `segments`.
        """
        return [self._resample_segment(s) for s in segments]

    def _resample_segment(self, segment: Segment) -> Segment:
        audio = segment.audio

        if audio.sample_rate == self.sample_rate:
            resampled_audio = audio
        else:
            signal = audio.read()
            resampled_signal = resampy.resample(signal, audio.sample_rate, self.sample_rate, axis=1)
            resampled_audio = MemoryAudioBuffer(resampled_signal, sample_rate=self.sample_rate)

        resampled_segment = Segment(
            label=self.output_label,
            span=segment.span,
            audio=resampled_audio,
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(resampled_segment, self.description, [segment])

        return resampled_segment
