__all__ = [
    "get_text_documents",
    "get_text_document",
    "get_text",
    "get_path_hf_dummy_vocab",
    "PATH_DOCCANO_FILES",
    "PATH_E3C_CORPUS_FILES",
]

from pathlib import Path

from medkit.core.text import TextDocument

_PATH_TO_TEXT_DOCS = Path(__file__).parent / "data" / "text"
_PATH_HF_DUMMY_VOCAB = Path(__file__).parent / "data" / "dummy_hf_vocab" / "vocab.txt"
PATH_DOCCANO_FILES = Path(__file__).parent / "data" / "doccano"
PATH_E3C_CORPUS_FILES = Path(__file__).parent / "data" / "e3c_corpus"


def get_path_hf_dummy_vocab():
    return _PATH_HF_DUMMY_VOCAB


def get_text_documents():
    return TextDocument.from_dir(_PATH_TO_TEXT_DOCS)


def get_text_document(name):
    return TextDocument.from_file(_PATH_TO_TEXT_DOCS / (name + ".txt"))


def get_text(name):
    path = _PATH_TO_TEXT_DOCS / (name + ".txt")
    with Path(path).open() as fp:
        return fp.read()
