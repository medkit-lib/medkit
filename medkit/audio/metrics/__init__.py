__all__ = []

from medkit.core.utils import modules_are_available

if modules_are_available(["pyannote"]) and modules_are_available(
    ["pyannote.core", "pyannote.metrics"]
):
    __all__.append("diarization")

if modules_are_available(["speechbrain"]):
    __all__.append("transcription")
