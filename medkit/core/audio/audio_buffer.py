from __future__ import annotations

__all__ = [
    "AudioBuffer",
    "FileAudioBuffer",
    "MemoryAudioBuffer",
    "PlaceholderAudioBuffer",
]

import abc
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import soundfile as sf
from typing_extensions import Self

from medkit.core import dict_conv


class AudioBuffer(abc.ABC, dict_conv.SubclassMapping):
    """Audio buffer base class. Gives access to raw audio samples."""

    @abc.abstractmethod
    def __init__(self, sample_rate: int, nb_samples: int, nb_channels: int):
        """
        Parameters
        ----------
        sample_rate:
            Sample rate of the signal, in samples per second.
        nb_samples:
            Duration of the signal in samples.
        nb_channels:
            Number of channels in the signal.
        """
        self.sample_rate = sample_rate
        self.nb_samples = nb_samples
        self.nb_channels = nb_channels

    @property
    def duration(self) -> float:
        """Duration of the signal in seconds."""
        return self.nb_samples / self.sample_rate

    @abc.abstractmethod
    def read(self, copy: bool = False) -> np.ndarray:
        """Return the signal in the audio buffer.

        Parameters
        ----------
        copy:
            If `True`, the returned array will be a copy that can be safely mutated.

        Returns
        -------
        np.ndarray:
            Raw audio samples
        """
        pass

    @abc.abstractmethod
    def trim(self, start: Optional[int], end: Optional[int]) -> AudioBuffer:
        """Return a new audio buffer pointing to portion of the signal in the original buffer,
        using boundaries in samples.

        Parameters
        ----------
        start:
            Start sample of the new buffer (defaults to `0`).
        end:
            End sample of the new buffer, excluded (default to full duration).

        Returns
        -------
        AudioBuffer:
            Trimmed audio buffer with new start and end samples, of same type as
            original audio buffer.
        """
        pass

    def trim_duration(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> AudioBuffer:
        """Return a new audio buffer pointing to a portion of the signal in the original buffer,
        using boundaries in seconds. Since `start_time` and `end_time` are in seconds, the exact
        trim boundaries will be rounded to the nearest sample and will therefore depend on the sampling
        rate.

        Parameters
        ----------
        start_time:
            Start time of the new buffer (defaults to `0.0`).
        end_time:
            End time of thew new buffer, excluded (default to full duration).

        Returns
        -------
        AudioBuffer:
            Trimmed audio buffer with new start and end samples, of same type as
            original audio buffer.
        """
        assert end_time is None or end_time <= self.duration
        start = round(start_time * self.sample_rate) if start_time is not None else None
        end = min(round(end_time * self.sample_rate), self.nb_samples) if end_time is not None else None
        return self.trim(start, end)

    def __init_subclass__(cls):
        AudioBuffer.register_subclass(cls)
        super().__init_subclass__()

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> Self:
        subclass = cls.get_subclass_for_data_dict(data_dict)
        if subclass is None:
            raise NotImplementedError(
                "AudioBuffer is an abstract class. Its class method `from_dict` is"
                " only used for calling the correct subclass `from_dict`."
            )

        return subclass.from_dict(data_dict)

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class FileAudioBuffer(AudioBuffer):
    """
    Audio buffer giving access to audio files stored on the filesystem (to use
    when manipulating unmodified raw audio).

    Supports all file formats handled by `libsndfile`
    (http://www.mega-nerd.com/libsndfile/#Features)
    """

    def __init__(
        self,
        path: Union[str, Path],
        trim_start: Optional[int] = None,
        trim_end: Optional[int] = None,
        sf_info: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        path:
            Path to the audio file.
        trim_start:
            First sample of audio file to consider.
        trim_end:
            First sample of audio file to exclude.
        sf_info:
            Optional metadata dict returned by soundfile.
        """
        path = Path(path)
        if sf_info is None:
            sf_info = sf.info(path)

        assert trim_start is None or 0 <= trim_start <= sf_info.frames
        assert trim_end is None or 0 <= trim_end <= sf_info.frames

        if trim_start is None:
            trim_start = 0
        if trim_end is None:
            trim_end = sf_info.frames

        sample_rate = sf_info.samplerate
        nb_samples = trim_end - trim_start
        nb_channels = sf_info.channels

        super().__init__(sample_rate=sample_rate, nb_samples=nb_samples, nb_channels=nb_channels)

        self.path = path
        self._trim_end = trim_end
        self._trim_start = trim_start
        self._sf_info = sf_info

    def read(self, copy: bool = False) -> np.ndarray:
        signal, _ = sf.read(
            self.path,
            start=self._trim_start,
            stop=self._trim_end,
            always_2d=True,
            dtype=np.float32,
        )
        return signal.T

    def trim(self, start: Optional[int] = None, end: Optional[int] = None) -> AudioBuffer:
        assert start is None or 0 <= start <= self.nb_samples
        assert end is None or 0 <= end <= self.nb_samples

        if start is not None:
            new_trim_start = self._trim_start + start
        else:
            new_trim_start = self._trim_start
        if end is not None:
            new_trim_end = self._trim_start + end
        else:
            new_trim_end = self._trim_end
        assert new_trim_start <= new_trim_end
        return FileAudioBuffer(self.path, new_trim_start, new_trim_end, self._sf_info)

    def to_dict(self) -> Dict[str, Any]:
        buffer_dict = dict(
            path=str(self.path),
            trim_start=self._trim_start,
            trim_end=self._trim_end,
        )
        dict_conv.add_class_name_to_data_dict(self, buffer_dict)
        return buffer_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(path=data["path"], trim_start=data["trim_start"], trim_end=data["trim_end"])

    def __eq__(self, other: object) -> bool:
        if type(other) is not self.__class__:
            return False
        return self.path == other.path and self._trim_end == other._trim_end and self._trim_start == other._trim_start


class MemoryAudioBuffer(AudioBuffer):
    """Audio buffer giving access to signals stored in memory
    (to use when reading/writing a modified audio signal)."""

    def __init__(self, signal: np.ndarray, sample_rate: int):
        """
        Parameters
        ----------
        signal:
            Samples constituting the audio signal, with shape `(nb_channel, nb_samples)`.
        sample_rate:
            Sample rate of the signal, in samples per second.
        """
        nb_channels, nb_samples = signal.shape

        super().__init__(sample_rate=sample_rate, nb_samples=nb_samples, nb_channels=nb_channels)

        self._signal = signal

    def read(self, copy: bool = False) -> np.ndarray:
        if copy:
            return self._signal.copy()
        else:
            return self._signal

    def trim(self, start: Optional[int] = None, end: Optional[int] = None) -> AudioBuffer:
        assert start is None or 0 <= start <= self.nb_samples
        assert end is None or 0 <= end <= self.nb_samples

        if start is None:
            start = 0
        if end is None:
            end = self.nb_samples
        assert start <= end
        return MemoryAudioBuffer(self._signal[:, start:end], self.sample_rate)

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("MemoryBuffer can't be converted to dict")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        raise NotImplementedError("MemoryBuffer can't be instantiated from dict")

    def __eq__(self, other: object) -> bool:
        if type(other) is not self.__class__:
            return False
        return np.array_equal(self._signal, other._signal)


class PlaceholderAudioBuffer(AudioBuffer):
    """Placeholder representing a MemoryAudioBuffer for which we have lost the actual signal.

    This class is only here so that MemoryAudioBuffer objects can be converted
    into json/yaml serializable dicts and then unserialized, but no further
    processing can be performed since the actual signal is not saved. Calling
    :meth`~read()` or :meth`~.trim()` will raise.
    """

    def __init__(self, sample_rate: int, nb_samples: int, nb_channels: int):
        super().__init__(sample_rate, nb_samples, nb_channels)

    @classmethod
    def from_audio_buffer(cls, audio_buffer: AudioBuffer) -> PlaceholderAudioBuffer:
        return cls(
            sample_rate=audio_buffer.sample_rate,
            nb_samples=audio_buffer.nb_samples,
            nb_channels=audio_buffer.nb_channels,
        )

    def read(self, copy: bool = False) -> np.ndarray:
        raise NotImplementedError("Cannot call read() on a PlaceholderAudioBuffer, signal is unknown")

    def trim(self, start: Optional[int], end: Optional[int]) -> AudioBuffer:
        raise NotImplementedError("Cannot call trim() on a PlaceholderAudioBuffer, signal is unknown")

    def to_dict(self) -> Dict[str, Any]:
        buffer_dict = dict(
            sample_rate=self.sample_rate,
            nb_samples=self.nb_samples,
            nb_channels=self.nb_channels,
        )
        dict_conv.add_class_name_to_data_dict(self, buffer_dict)
        return buffer_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
            sample_rate=data["sample_rate"],
            nb_samples=data["nb_samples"],
            nb_channels=data["nb_channels"],
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not self.__class__:
            return False
        return (
            self.sample_rate == other.sample_rate
            and self.nb_samples == other.nb_samples
            and self.nb_channels == other.nb_channels
        )
