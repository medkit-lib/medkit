from __future__ import annotations

__all__ = ["Downmixer"]

import numpy as np

from medkit.core.audio import MemoryAudioBuffer, PreprocessingOperation, Segment


class Downmixer(PreprocessingOperation):
    """Downmixing operation converting multichannel audio signals to mono."""

    def __init__(
        self,
        output_label: str,
        prevent_clipping: bool = True,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        output_label : str
            Label of output downmixed segments.
        prevent_clipping : bool, default=True
            If `True`, normalize downmixed signals by number of channels to prevent clipping.
        uid : str, optional
            Identifier of the downmixer.
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        self.prevent_clipping = prevent_clipping

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Return a downmixed segment for each segment in `segments`.

        Parameters
        ----------
        segments : list of Segment
            Audio segments to downmix.

        Returns
        -------
        list of Segment
            Downmixed segments, one per segment in `segments`.
        """
        return [self._downmix_segment(s) for s in segments]

    def _downmix_segment(self, segment: Segment) -> Segment:
        audio = segment.audio
        if segment.audio.nb_channels == 1:
            downmixed_audio = audio
        else:
            signal = segment.audio.read()
            downmixed_signal = np.sum(signal, axis=0, keepdims=True)
            if self.prevent_clipping:
                downmixed_signal /= signal.shape[0]
            downmixed_audio = MemoryAudioBuffer(downmixed_signal, sample_rate=audio.sample_rate)

        downmixed_segment = Segment(
            label=self.output_label,
            span=segment.span,
            audio=downmixed_audio,
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(downmixed_segment, self.description, [segment])

        return downmixed_segment
