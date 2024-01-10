import pytest

from medkit.io.medkit_json import load_text_documents
from medkit.tools.e3c_corpus import (
    convert_data_annotation_to_medkit,
    convert_data_collection_to_medkit,
    load_data_annotation,
    load_data_collection,
)
from tests.data_utils import PATH_E3C_CORPUS_FILES


@pytest.fixture(scope="module")
def e3c_corpus_path():
    return PATH_E3C_CORPUS_FILES


def test_convert_data_collection_to_medkit(e3c_corpus_path, tmpdir):
    medkit_file = tmpdir / "medkit.jsonl"
    convert_data_collection_to_medkit(dir_path=e3c_corpus_path, output_file=medkit_file)
    docs_from_corpus = load_data_collection(dir_path=e3c_corpus_path)
    docs_from_medkit = load_text_documents(medkit_file)
    assert next(iter(docs_from_corpus)) == next(iter(docs_from_medkit))


def test_convert_data_annotation_to_medkit(e3c_corpus_path, tmpdir):
    medkit_file = tmpdir / "medkit.jsonl"
    convert_data_annotation_to_medkit(dir_path=e3c_corpus_path, output_file=medkit_file, keep_sentences=True)
    docs_from_corpus = list(load_data_annotation(dir_path=e3c_corpus_path, keep_sentences=True))
    docs_from_medkit = list(load_text_documents(medkit_file))

    assert docs_from_corpus == docs_from_medkit
