"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[webrtc-voice-detector]`.
"""
from __future__ import annotations

__all__ = ["WebRTCVoiceDetector"]

import collections
from typing import Iterator

import numpy as np
import webrtcvad
from typing_extensions import Literal

from medkit.core.audio import Segment, SegmentationOperation, Span

_SUPPORTED_SAMPLE_RATES = {8000, 16000, 32000, 48000}


class WebRTCVoiceDetector(SegmentationOperation):
    """Voice Activity Detection operation relying on the `webrtcvad` package.

    Per-frame VAD results of `webrtcvad` are aggregated with a switch algorithm
    considering the percentage of speech/non-speech frames in a wider sliding window.

    Input segments must be mono at 8kHZ, 16kHz, 32kHz or 48Khz.
    """

    def __init__(
        self,
        output_label: str,
        aggressiveness: Literal[0, 1, 2, 3] = 2,
        frame_duration: Literal[10, 20, 30] = 30,
        nb_frames_in_window: int = 10,
        switch_ratio: float = 0.9,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        output_label : str
            Label of output speech segments.
        aggressiveness : {0, 1, 2, 3}, default=2
            Aggressiveness param passed to `webrtcvad` (the higher, the more likely
            to detect speech).
        frame_duration : {10, 20, 30}, default=30
            Duration in milliseconds of frames passed to `webrtcvad`.
        nb_frames_in_window : int, default=10
            Number of frames in the sliding window used when aggregating per-frame VAD
            results.
        switch_ratio : float, default=0.9
            Percentage of speech/non-speech frames required to switch the window speech
            state when aggregating per-frame VAD results.
        uid : str, optional
            Identifier of the detector.
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        self.aggressiveness = aggressiveness
        self.frame_duration = frame_duration
        self.nb_frames_in_window = nb_frames_in_window
        self.switch_ratio = switch_ratio

        self._vad = webrtcvad.Vad(aggressiveness)

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Return all speech segments detected for all input `segments`.

        Parameters
        ----------
        segments : list of Segment
            Audio segments on which to perform VAD.

        Returns
        -------
        list of Segment
            Segments detected as containing speech activity.
        """
        return [voice_seg for seg in segments for voice_seg in self._detect_activity_in_segment(seg)]

    def _detect_activity_in_segment(self, segment: Segment) -> Iterator[Segment]:
        audio = segment.audio
        if audio.nb_channels > 1:
            msg = f"Segment with identifier {segment.uid} has multi-channel audio, which is not supported"
            raise RuntimeError(msg)
        if audio.sample_rate not in _SUPPORTED_SAMPLE_RATES:
            msg = f"Segment with identifier {segment.uid} has non-supported sample rate {audio.sample_rate}"
            raise RuntimeError(msg)

        sample_rate = audio.sample_rate
        nb_samples = audio.nb_samples

        signal = audio.read()
        # convert float32 signal to int16 (required by webrtcvad)
        int_signal = (signal * 32767).astype(np.int16)
        frame_length = int(self.frame_duration * sample_rate / 1000)
        # zero-pad tail and split in frames
        padding_length = frame_length - nb_samples % frame_length
        padding = np.zeros((1, padding_length), dtype=np.int16)
        int_signal = np.concatenate((int_signal, padding), axis=1)
        frames = np.split(int_signal, int_signal.shape[1] / frame_length, axis=1)

        # run vad
        speech_frame_indices = self._get_aggregated_vad(frames, sample_rate)

        # generate segments
        for start_frame_index, end_frame_index in speech_frame_indices:
            # trim original audio
            start = start_frame_index * frame_length
            end = end_frame_index * frame_length
            end = min(end, nb_samples)
            voiced_audio = audio.trim(start, end)
            start_time = start / sample_rate
            end_time = end / sample_rate
            # build corresponding span
            voiced_span = Span(segment.span.start + start_time, segment.span.start + end_time)
            voiced_segment = Segment(label=self.output_label, span=voiced_span, audio=voiced_audio)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(voiced_segment, self.description, [segment])

            yield voiced_segment

    # from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    def _get_aggregated_vad(self, frames, sample_rate):
        """Return index ranges of voiced frames using webrtcvad"""
        # deque for our sliding window ring buffer
        window_ring_buffer = collections.deque(maxlen=self.nb_frames_in_window)
        # we have two states: SPEECH and NONSPEECH (we start in NONSPEECH)
        is_speech = False

        speech_frame_ranges = []
        start_index = None
        for i, frame in enumerate(frames):
            # compute speech state for frame and push it to ring buffer of frames in window
            frame_bytes = frame.tobytes()
            frame_is_speech = self._vad.is_speech(frame_bytes, sample_rate)
            window_ring_buffer.append((i, frame_is_speech))

            if not is_speech:
                nb_speech_frames = sum(frame_is_speech for _, frame_is_speech in window_ring_buffer)
                # if we are NONSPEECH and more than 90% of the frames in
                # the ring buffer are speech frames, then enter the
                # SPEECH state
                if nb_speech_frames > self.switch_ratio * window_ring_buffer.maxlen:
                    is_speech = True
                    # all frames in the ring buffer are retrospectively considered as SPEECH
                    start_index, _ = window_ring_buffer[0]
                    # all upcoming frames will also be considered SPEECH until we enter
                    # NONSPEECH state
                    window_ring_buffer.clear()
            else:
                nb_non_speech_frames = sum(not frame_is_speech for _, frame_is_speech in window_ring_buffer)
                # if more than 90% of the frames in the ring buffer are
                # non-speech, then enter NONSPEECH
                if nb_non_speech_frames > self.switch_ratio * window_ring_buffer.maxlen:
                    is_speech = False
                    # push indices of the SPEECH range that just ended
                    end_index, _ = window_ring_buffer[-1]
                    speech_frame_ranges.append((start_index, end_index))
                    window_ring_buffer.clear()

        # handle trail
        if is_speech and window_ring_buffer:
            end_index, _ = window_ring_buffer[-1]
            speech_frame_ranges.append((start_index, end_index))

        return speech_frame_ranges
