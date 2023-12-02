from pathlib import Path

from medkit.core import Collection
from medkit.core.audio import AudioDocument, FileAudioBuffer
from medkit.core.text import TextDocument

_AUDIO_FILE = Path("tests/data/audio/voice.ogg")


def test_basic():
    text_doc_1 = TextDocument(text="Hello")
    text_doc_2 = TextDocument(text="Hi")
    audio_doc = AudioDocument(audio=FileAudioBuffer(path=_AUDIO_FILE))

    collection = Collection(text_docs=[text_doc_1, text_doc_2], audio_docs=[audio_doc])
    assert collection.text_docs == [text_doc_1, text_doc_2]
    assert collection.audio_docs == [audio_doc]
    assert collection.all_docs == [text_doc_1, text_doc_2, audio_doc]
