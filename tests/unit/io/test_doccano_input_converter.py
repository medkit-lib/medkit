import logging
from zipfile import ZipFile

import pytest

from medkit.core.prov_tracer import ProvTracer
from medkit.core.text.span import Span
from medkit.io import DoccanoInputConverter, DoccanoTask
from tests.data_utils import PATH_DOCCANO_FILES


def create_doccano_zip_files_disk(tmp_path, filename):
    dir_path = tmp_path / filename
    dir_path.mkdir(exist_ok=True)

    with ZipFile(dir_path / "file.zip", "w") as zip_file:
        filepath = PATH_DOCCANO_FILES / f"{filename}.jsonl"
        zip_file.write(filepath)


def test_relation_extraction_converter(tmp_path):
    task = DoccanoTask.RELATION_EXTRACTION
    create_doccano_zip_files_disk(tmp_path, filename=task.value)
    expected_metadata = {"custom_metadata": "custom", "doc_id": 1234}

    converter = DoccanoInputConverter(task=task)
    documents = converter.load_from_directory_zip(dir_path=f"{tmp_path}/{task.value}")
    assert len(documents) == 1

    document = documents[0]
    assert len(document.anns.entities) == 2
    assert len(document.anns.relations) == 1
    assert document.metadata == expected_metadata

    entity_0 = document.anns.get(label="ORG")[0]
    entity_1 = document.anns.get(label="DATE")[0]
    relation = document.anns.get(label="created_in")[0]

    assert entity_0.text == "medkit"
    assert relation.metadata["doccano_id"] == 0
    assert relation.source_id == entity_0.uid
    assert relation.target_id == entity_1.uid


def test_sequence_labeling_converter(tmp_path):
    task = DoccanoTask.SEQUENCE_LABELING
    create_doccano_zip_files_disk(tmp_path, filename=task.value)

    converter = DoccanoInputConverter(task=task)
    documents = converter.load_from_directory_zip(dir_path=f"{tmp_path}/{task.value}")
    assert len(documents) == 1

    document = documents[0]
    assert len(document.anns.entities) == 2
    assert len(document.anns.relations) == 0
    assert document.metadata == {}

    entity_0 = document.anns.get(label="ORG")[0]
    assert entity_0.text == "medkit"


def test_text_classification_converter(tmp_path):
    task = DoccanoTask.TEXT_CLASSIFICATION
    create_doccano_zip_files_disk(tmp_path, filename=task.value)

    converter = DoccanoInputConverter(task=task)
    documents = converter.load_from_directory_zip(dir_path=f"{tmp_path}/{task.value}")
    assert len(documents) == 1

    document = documents[0]
    assert len(document.anns.entities) == 0
    assert len(document.anns.relations) == 0
    assert document.metadata == {}

    expected_label = converter.attr_label
    attrs = document.attrs.get(label=expected_label)
    assert len(attrs) == 1
    assert attrs[0].value == "header"


def test_crlf_character(tmp_path, caplog):
    # test when doccano export a document from a project with
    # 'count grapheme clusters as one character'
    # This option defines the span differently in Doccano
    # In medkit this results in alignment problems.
    task = DoccanoTask.RELATION_EXTRACTION
    filename = "relation_extraction_wrong_character"
    create_doccano_zip_files_disk(tmp_path, filename=filename)

    # test default config
    with caplog.at_level(logging.WARNING, logger="medkit.io.doccano"):
        converter = DoccanoInputConverter(task=task)
        documents = converter.load_from_directory_zip(dir_path=f"{tmp_path}/{filename}")
        assert "1/1 documents contain" in caplog.text

    document = documents[0]
    assert len(document.anns.entities) == 2
    # in doccano the text is '2020'
    entity_no_aligned = document.anns.get(label="DATE")[0]
    assert entity_no_aligned.text == " 202"
    assert entity_no_aligned.spans == [Span(22, 26)]


TEST_PROV_BY_TASK = [
    (DoccanoTask.RELATION_EXTRACTION, True),
    (DoccanoTask.SEQUENCE_LABELING, True),
    (DoccanoTask.TEXT_CLASSIFICATION, False),
]


@pytest.mark.parametrize(
    ("task", "check_prov_entity"),
    TEST_PROV_BY_TASK,
)
def test_prov(tmp_path, task, check_prov_entity):
    converter = DoccanoInputConverter(task=task)
    prov_tracer = ProvTracer()
    converter.set_prov_tracer(prov_tracer)

    create_doccano_zip_files_disk(tmp_path, filename=task.value)
    docs = converter.load_from_directory_zip(dir_path=f"{tmp_path}/{task.value}")

    doc = docs[0]
    ann = doc.anns.entities[0] if check_prov_entity else doc.attrs.get()[0]
    prov = prov_tracer.get_prov(ann.uid)
    assert prov.data_item == ann
    assert prov.op_desc == converter.description
    assert len(prov.source_data_items) == 0


def test_exceptions(tmp_path):
    # testing incoherence between data and task
    task = DoccanoTask.SEQUENCE_LABELING
    wrong_task = DoccanoTask.RELATION_EXTRACTION
    create_doccano_zip_files_disk(tmp_path, filename=wrong_task.value)

    with pytest.raises(ValueError, match="Impossible to convert"):
        DoccanoInputConverter(task=task).load_from_directory_zip(dir_path=f"{tmp_path}/{wrong_task.value}")

    # testing incoherence between data and task
    task = DoccanoTask.RELATION_EXTRACTION
    wrong_task = DoccanoTask.SEQUENCE_LABELING
    create_doccano_zip_files_disk(tmp_path, filename=wrong_task.value)

    with pytest.raises(ValueError, match="Impossible to convert"):
        DoccanoInputConverter(task=task).load_from_directory_zip(dir_path=f"{tmp_path}/{wrong_task.value}")

    # testing incoherence between data and task
    task = DoccanoTask.TEXT_CLASSIFICATION
    wrong_task = DoccanoTask.SEQUENCE_LABELING
    create_doccano_zip_files_disk(tmp_path, filename=wrong_task.value)

    with pytest.raises(ValueError, match="Impossible to convert"):
        DoccanoInputConverter(task=task).load_from_directory_zip(dir_path=f"{tmp_path}/{wrong_task.value}")
