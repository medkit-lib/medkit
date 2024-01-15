import pytest

pytest.importorskip(modname="transformers", reason="transformers is not installed")

from transformers import BertTokenizerFast

from medkit.core.text import Entity, Span, TextDocument
from medkit.text.ner.hf_tokenization_utils import (
    SPECIAL_TAG_ID_HF,
    align_and_map_tokens_with_tags,
    convert_labels_to_tags,
    transform_entities_to_tags,
)
from tests.data_utils import get_path_hf_dummy_vocab


def _get_document():
    return TextDocument(
        text="medkit is a python library",
        anns=[
            Entity(
                label="corporation",
                spans=[Span(start=0, end=6)],
                text="medkit",
            ),
            Entity(
                label="language",
                spans=[Span(start=12, end=18)],
                text="python",
            ),
        ],
    )


@pytest.fixture()
def tokenizer():
    return BertTokenizerFast(get_path_hf_dummy_vocab())


TEST_CONFIG = (
    (
        "bilou",
        ["O", "B-corporation", "L-corporation", "O", "O", "U-language", "O", "O"],
    ),
    (
        "iob2",
        ["O", "B-corporation", "I-corporation", "O", "O", "B-language", "O", "O"],
    ),
)


@pytest.mark.parametrize(
    ("tagging_scheme", "expected_tags"),
    TEST_CONFIG,
    ids=[
        "transform_bilou",
        "transform_iob2",
    ],
)
def test_transform_entities_offset(tokenizer, tagging_scheme, expected_tags):
    # Testing transformation changing tagging scheme
    document = _get_document()
    text_encoding = tokenizer(document.text).encodings[0]
    tags = transform_entities_to_tags(
        text_encoding=text_encoding,
        entities=document.anns.entities,
        tagging_scheme=tagging_scheme,
    )
    assert tags == expected_tags


def test_aligned_tokens_with_tags(tokenizer):
    text_encoding = tokenizer("medkit").encodings[0]
    tag_mapping = {"O": 0, "B-corporation": 1, "L-corporation": 2}

    tags_to_aligned = ["O", "B-corporation", "L-corporation", "O"]
    # tag all subtokens by word
    tags_ids = align_and_map_tokens_with_tags(
        text_encoding=text_encoding,
        tags=tags_to_aligned,
        tag_to_id=tag_mapping,
        map_sub_tokens=True,
    )
    assert tags_ids == [SPECIAL_TAG_ID_HF, 1, 2, SPECIAL_TAG_ID_HF]

    # tag only the first token by word, recommended
    tags_ids = align_and_map_tokens_with_tags(
        text_encoding=text_encoding,
        tags=tags_to_aligned,
        tag_to_id=tag_mapping,
        map_sub_tokens=False,
    )
    assert tags_ids == [SPECIAL_TAG_ID_HF, 1, SPECIAL_TAG_ID_HF, SPECIAL_TAG_ID_HF]


def test_convert_labels_to_tags():
    tags = convert_labels_to_tags(labels=["procedure"], tagging_scheme="bilou")
    assert tags == {
        "O": 0,
        "B-procedure": 1,
        "I-procedure": 2,
        "L-procedure": 3,
        "U-procedure": 4,
    }
    tags = convert_labels_to_tags(labels=["procedure"], tagging_scheme="iob2")
    assert tags == {
        "O": 0,
        "B-procedure": 1,
        "I-procedure": 2,
    }
