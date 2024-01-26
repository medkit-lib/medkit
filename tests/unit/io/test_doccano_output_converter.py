import dataclasses
import json
import logging

import pytest

from medkit.core import Attribute
from medkit.core.text import Entity, Relation, Segment, Span, TextDocument
from medkit.io import DoccanoOutputConverter, DoccanoTask
from tests.data_utils import PATH_DOCCANO_FILES

_METADATA = {"custom_metadata": "custom", "doc_id": 1234}


# mock of UUID class used by generate_deterministic_id
@dataclasses.dataclass()
class _MockUUID:
    int: int


@pytest.fixture(scope="module", autouse=True)
def _mocked_generator_id(module_mocker):
    # mock doccano IDs, force to 0 or 1 i.e. e0->0
    module_mocker.patch(
        "medkit.io.doccano.generate_deterministic_id",
        lambda ref_id: _MockUUID(int(ref_id[1])),
    )


def _get_doc_by_task(task: DoccanoTask):
    # get a TextDocument by task, seqlabeling and RelationExtraction use
    # the same doc, the output format changes
    doc = TextDocument(text="medkit was created in 2022")

    if task in (DoccanoTask.RELATION_EXTRACTION, DoccanoTask.SEQUENCE_LABELING):
        medkit_anns = [
            Entity(label="ORG", spans=[Span(0, 6)], text="medkit", uid="e0"),
            Entity(label="DATE", spans=[Span(22, 26)], text="2022", uid="e1"),
            Relation(label="created_in", source_id="e0", target_id="e1", uid="r0"),
            Segment(label="sentence", text="medkit was created in 2022", spans=[Span(0, 26)]),
        ]

        for ann in medkit_anns:
            doc.anns.add(ann)

    elif task == DoccanoTask.TEXT_CLASSIFICATION:
        attr = Attribute(label="category", value="header")
        doc.attrs.add(attr)

    # only RELATION EXTRACTION has metadata
    if task == DoccanoTask.RELATION_EXTRACTION:
        doc.metadata = _METADATA
    return doc


def _load_json_file(filepath):
    from pathlib import Path

    with Path(filepath).open() as fp:
        return json.load(fp)


@pytest.mark.parametrize(
    "task",
    [
        DoccanoTask.RELATION_EXTRACTION,
        DoccanoTask.TEXT_CLASSIFICATION,
        DoccanoTask.SEQUENCE_LABELING,
    ],
)
def test_save_by_task_with_metadat(tmp_path, task):
    medkit_docs = [_get_doc_by_task(task)]
    converter = DoccanoOutputConverter(task=task, attr_label="category", include_metadata=True)

    output_file = tmp_path / f"{task.value}.jsonl"
    converter.save(medkit_docs, output_file=output_file)

    # check generated data
    data = _load_json_file(output_file)
    expected_data = _load_json_file(PATH_DOCCANO_FILES / f"{task.value}.jsonl")
    assert data == expected_data


def test_save_segments(tmp_path):
    medkit_docs = [_get_doc_by_task(DoccanoTask.SEQUENCE_LABELING)]
    converter = DoccanoOutputConverter(
        task=DoccanoTask.SEQUENCE_LABELING,
        anns_labels="sentence",
        ignore_segments=False,
    )

    output_file = tmp_path / "segments.jsonl"
    converter.save(medkit_docs, output_file=output_file)

    # check generated data
    data = _load_json_file(output_file)
    expected_data = _load_json_file(PATH_DOCCANO_FILES / "segments.jsonl")
    assert data == expected_data


def test_warnings(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="medkit.io.doccano"):
        # get only ORG entities and created_in relations
        # target of the relation is missing
        task = DoccanoTask.RELATION_EXTRACTION
        converter = DoccanoOutputConverter(task=task, anns_labels=["ORG", "created_in"])
        medkit_docs = [_get_doc_by_task(task)]

        output_file = tmp_path / f"{task.value}.jsonl"
        converter.save(medkit_docs, output_file=output_file)
        assert "Entity source/target was no found" in caplog.text

    with pytest.raises(KeyError, match="The attribute with the corresponding .*"):  # noqa: PT012
        # the attr_label is header not is_negated
        task = DoccanoTask.TEXT_CLASSIFICATION
        medkit_docs = [_get_doc_by_task(task)]
        converter = DoccanoOutputConverter(task=task, attr_label="is_negated")

        output_file = tmp_path / f"{task.value}.jsonl"
        converter.save(medkit_docs, output_file=output_file)


@pytest.mark.parametrize(
    "task",
    [
        DoccanoTask.RELATION_EXTRACTION,
        DoccanoTask.TEXT_CLASSIFICATION,
        DoccanoTask.SEQUENCE_LABELING,
    ],
)
def test_save_by_task_without_metadata(tmp_path, task):
    medkit_docs = [_get_doc_by_task(task)]
    converter = DoccanoOutputConverter(task=task, attr_label="category", include_metadata=False)

    output_file = tmp_path / f"{task.value}.jsonl"
    converter.save(medkit_docs, output_file=output_file)

    # check generated data
    data = _load_json_file(output_file)
    expected_data = _load_json_file(PATH_DOCCANO_FILES / f"{task.value}.jsonl")

    # Prepare expected_data, the test forces not to export metadata
    # the data should not include it
    for key in _METADATA:
        expected_data.pop(key, None)

    assert data == expected_data
