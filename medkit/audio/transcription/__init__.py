from medkit.audio.transcription.doc_transcriber import DocTranscriber, TranscriptionOperation
from medkit.audio.transcription.transcribed_text_document import TranscribedTextDocument
from medkit.core.utils import modules_are_available

__all__ = [
    "DocTranscriber",
    "TranscriptionOperation",
    "TranscribedTextDocument",
]

if modules_are_available(["torchaudio", "transformers"]):
    from medkit.audio.transcription.hf_transcriber import HFTranscriber

    __all__ += ["HFTranscriber"]

if modules_are_available(["torch", "speechbrain"]):
    from medkit.audio.transcription.sb_transcriber import SBTranscriber

    __all__ += ["SBTranscriber"]
