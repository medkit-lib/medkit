from pathlib import Path

from medkit.io import medkit_json
from tests.unit.io.medkit_json._text_common import (
    ANNS_JSONL_FILE,
    DOC_JSON_FILE,
    DOCS_JSONL_FILE,
    SPLIT_DOC_ANNS_JSONL_FILE,
    SPLIT_DOC_JSON_FILE,
    build_anns,
    build_doc,
    build_docs,
)


def _check_json_files_are_equal(json_file: Path, expected_json_file: Path):
    json_content = json_file.read_text().split("\n")
    expected_json_content = expected_json_file.read_text().split("\n")
    assert json_content == expected_json_content


def test_save_document(tmp_path):
    doc = build_doc()

    output_file = tmp_path / "doc.json"
    medkit_json.save_text_document(doc, output_file)

    _check_json_files_are_equal(output_file, DOC_JSON_FILE)


def test_save_documents(tmp_path):
    docs = build_docs()

    output_file = tmp_path / "docs.jsonl"
    medkit_json.save_text_documents(docs, output_file)

    _check_json_files_are_equal(output_file, DOCS_JSONL_FILE)


def test_save_anns(tmp_path):
    anns = build_anns()

    output_file = tmp_path / "anns.jsonl"
    medkit_json.save_text_anns(anns, output_file)

    _check_json_files_are_equal(output_file, ANNS_JSONL_FILE)


def test_save_document_split(tmp_path):
    doc = build_doc()

    output_file = tmp_path / "split_doc.json"
    anns_output_file = tmp_path / "split_doc_anns.jsonl"
    medkit_json.save_text_document(doc, output_file, split_anns=True, anns_output_file=anns_output_file)

    _check_json_files_are_equal(output_file, SPLIT_DOC_JSON_FILE)
    _check_json_files_are_equal(anns_output_file, SPLIT_DOC_ANNS_JSONL_FILE)
