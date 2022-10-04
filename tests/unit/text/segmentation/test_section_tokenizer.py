import pytest

from medkit.core import ProvTracer
from medkit.core.text import Span, Segment
from medkit.text.segmentation.section_tokenizer import (
    SectionTokenizer,
    SectionModificationRule,
)
import tests.data_utils as data_utils

TEST_CONFIG = [
    (
        "eds/clean/cas1",
        [
            ([Span(start=0, end=418)], "antecedent"),
            ([Span(start=418, end=1231)], "mode_de_vie"),
            ([Span(start=1231, end=1606)], "conclusion"),
            ([Span(start=1606, end=1899)], "conclusion"),
            ([Span(start=1899, end=2268)], "conclusion"),
            ([Span(start=2268, end=3109)], "conclusion"),
            ([Span(start=3109, end=3246)], "antecedent"),
            ([Span(start=3246, end=3843)], "antecedent"),
            ([Span(start=3843, end=3916)], "antecedent"),
            ([Span(start=3916, end=6668)], "antecedent"),
            ([Span(start=6668, end=9094)], "conclusion"),
        ],
    ),
    ("eds/clean/cas2", [([Span(start=0, end=6671)], "head")]),
    (
        "eds/clean/cas3",
        [
            ([Span(start=0, end=315)], "head"),
            ([Span(start=315, end=369)], "examen_clinique"),
        ],
    ),
]


def _get_clean_text_segment(filepath):
    text = data_utils.get_text(filepath)
    return Segment(
        label="clean_text",
        spans=[Span(0, len(text))],
        text=text,
    )


@pytest.mark.parametrize("filepath,expected_sections", TEST_CONFIG)
def test_run(filepath, expected_sections):
    clean_text_segment = _get_clean_text_segment(filepath)

    section_tokenizer = SectionTokenizer.get_example()
    sections = section_tokenizer.run([clean_text_segment])

    assert len(sections) == len(expected_sections)
    for i, (spans, attr_value) in enumerate(expected_sections):
        assert sections[i].spans == spans
        assert sections[i].metadata["name"] == attr_value


def test_run_with_rules():
    filepath = TEST_CONFIG[0][0]
    clean_text_segment = _get_clean_text_segment(filepath)

    section_dict = {"antecedent": ["Antécédents médicaux"], "examen": ["Examen :"]}
    section_rules = (
        SectionModificationRule(
            section_name="antecedent",
            new_section_name="antecedent_before_exam",
            order="BEFORE",
            other_sections=["examen"],
        ),
        SectionModificationRule(
            section_name="examen",
            new_section_name="exam_after_antecedent",
            order="AFTER",
            other_sections=["antecedent"],
        ),
    )
    section_tokenizer = SectionTokenizer(
        section_dict=section_dict, section_rules=section_rules
    )
    sections = section_tokenizer.run([clean_text_segment])

    assert len(sections) == 2
    sections_antecedent = [
        section
        for section in sections
        if section.metadata["name"] == "antecedent_before_exam"
    ]
    assert len(sections_antecedent) == 1
    section_examen = [
        section
        for section in sections
        if section.metadata["name"] == "exam_after_antecedent"
    ]
    assert len(section_examen) == 1


def test_prov():
    filepath = TEST_CONFIG[0][0]
    clean_text_segment = _get_clean_text_segment(filepath)

    section_dict = {"antecedent": ["Antécédents médicaux"], "examen": ["Examen :"]}
    tokenizer = SectionTokenizer(section_dict)
    prov_tracer = ProvTracer()
    tokenizer.set_prov_tracer(prov_tracer)
    sections = tokenizer.run([clean_text_segment])

    section_1 = sections[0]
    prov_1 = prov_tracer.get_prov(section_1.id)
    assert prov_1.data_item == section_1
    assert prov_1.op_desc == tokenizer.description
    assert prov_1.source_data_items == [clean_text_segment]

    section_2 = sections[1]
    prov_2 = prov_tracer.get_prov(section_2.id)
    assert prov_2.data_item == section_2
    assert prov_2.op_desc == tokenizer.description
    assert prov_2.source_data_items == [clean_text_segment]