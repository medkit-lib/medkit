from medkit.audio.transcription.doc_transcriber import DocTranscriber, TranscriptionOperation
from medkit.audio.transcription.transcribed_text_document import TranscribedTextDocument

__all__ = [
    "DocTranscriber",
    "TranscriptionOperation",
    "TranscribedTextDocument",
]

try:
    from medkit.audio.transcription.hf_transcriber import HFTranscriber

    __all__ += ["HFTranscriber"]
except ModuleNotFoundError:
    pass

try:
    from medkit.audio.transcription.sb_transcriber import SBTranscriber

    __all__ += ["SBTranscriber"]
except ModuleNotFoundError:
    pass
