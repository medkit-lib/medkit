__all__ = [
    "DocTranscriber",
    "TranscriptionOperation",
    "TranscribedTextDocument",
]

from medkit.core.utils import modules_are_available

from .doc_transcriber import DocTranscriber, TranscriptionOperation
from .transcribed_text_document import TranscribedTextDocument

if modules_are_available(["torchaudio", "transformers"]):
    __all__ += ["hf_transcriber"]

if modules_are_available(["torch", "speechbrain"]):
    __all__ += ["sb_transcriber"]
