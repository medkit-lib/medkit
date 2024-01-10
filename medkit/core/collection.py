__all__ = ["Collection"]

from typing import Any, Dict, List, Optional

from medkit.core.audio import AudioDocument
from medkit.core.document import Document
from medkit.core.text import TextDocument


class Collection:
    """
    Collection of documents of any modality (text, audio).

    This class allows to group together a set of documents representing a common
    unit (for instance a patient), even if they don't belong to the same modality.

    This class is still a work-in-progress. In the future it should be possible to attach
    additional information to a `Collection`.
    """

    def __init__(
        self,
        *,
        text_docs: Optional[List[TextDocument]] = None,
        audio_docs: Optional[List[AudioDocument]] = None,
    ):
        """
        Parameters
        -----------
        text_docs:
            List of text documents
        audio_docs:
            List of audio documents
        """

        if text_docs is None and audio_docs is None:
            raise ValueError("Collection must received at least one list of documents at init")

        if text_docs is None:
            text_docs = []
        if audio_docs is None:
            audio_docs = []

        self.text_docs = text_docs
        self.audio_docs = audio_docs

    @property
    def all_docs(self) -> List[Document]:
        """
        List of all the documents belonging to the document, whatever they
        modality
        """
        return self.text_docs + self.audio_docs

    def to_dict(self) -> Dict[str, Any]:
        text_docs = [d.to_dict() for d in self.text_docs]
        audio_docs = [d.to_dict() for d in self.audio_docs]
        return dict(text_docs=text_docs, audio_docs=audio_docs)
