from __future__ import annotations

import pytest

from medkit.core import ProvTracer
from medkit.core.text import Segment, Span, TextAnnotation, TextDocument
from medkit.core.text.operation import CustomTextOpType, create_text_operation


@pytest.fixture()
def init_data():
    text = "hello world"
    doc = TextDocument(text=text)
    words = text.split()
    for word in words:
        start = text.find(word)
        end = start + len(word)
        segment = Segment(label="word", spans=[Span(start, end)], text=word)
        doc.anns.add(segment)
    return doc, words


@pytest.mark.parametrize(
    "prov_enabled",
    [False, True],
)
def test_create_text_operation(init_data, prov_enabled):
    doc, words = init_data
    prov_tracer = ProvTracer() if prov_enabled else None

    def upper_anns(segment: Segment) -> Segment:
        text = segment.text.upper()
        return Segment(label="upper_word", spans=segment.spans, text=text)

    operation = create_text_operation(
        function=upper_anns,
        function_type=CustomTextOpType.CREATE_ONE_TO_N,
        name="upper_words",
    )
    assert operation.description.class_name == "_CustomTextOperation"
    assert operation.description.name == "upper_words"

    if prov_enabled:
        operation.set_prov_tracer(prov_tracer)

    segm_inputs = doc.anns.get()
    res = operation.run(segm_inputs)

    for index, segment in enumerate(res):
        assert segment.text == segm_inputs[index].text.upper()
        if prov_enabled:
            prov = prov_tracer.get_prov(segment.uid)
            assert prov.op_desc == operation.description
            assert prov.source_data_items == [segm_inputs[index]]


def test_create_text_operation_extract(init_data):
    doc, words = init_data

    def extract_annotations_from_doc(document: TextDocument) -> list[TextAnnotation]:
        return document.anns.get()

    operation = create_text_operation(
        name="extract_anns_from_doc",
        function=extract_annotations_from_doc,
        function_type=CustomTextOpType.EXTRACT_ONE_TO_N,
    )
    assert operation.description.class_name == "_CustomTextOperation"
    assert operation.description.name == "extract_anns_from_doc"

    res = operation.run([doc])

    for segment in res:
        assert segment.text in words


def test_create_text_operation_filter(init_data):
    doc, words = init_data

    def keep_hello_segment(segment: Segment) -> bool:
        if segment.text == "hello":
            return True
        return False

    anns = doc.anns.get()
    filter_ = create_text_operation(
        function=keep_hello_segment,
        function_type=CustomTextOpType.FILTER,
    )
    assert filter_.description.name == "keep_hello_segment"
    assert filter_.description.config.get("function_type") == CustomTextOpType.FILTER.name

    res: list[Segment] = filter_.run(anns)
    assert len(res) == 1
    assert res[0].text == "hello"
