import pytest

from medkit.core import ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.segmentation.section_tokenizer import (
    _PATH_TO_DEFAULT_RULES,
    SectionModificationRule,
    SectionTokenizer,
)
from tests import data_utils

TEST_CONFIG = [
    (
        "eds/clean/cas1",
        [
            ([Span(start=0, end=417)], "antecedent"),
            ([Span(start=418, end=1230)], "mode_de_vie"),
            ([Span(start=1231, end=1605)], "conclusion"),
            ([Span(start=1606, end=1898)], "conclusion"),
            ([Span(start=1899, end=2267)], "conclusion"),
            ([Span(start=2268, end=3108)], "conclusion"),
            ([Span(start=3109, end=3245)], "antecedent"),
            ([Span(start=3246, end=3842)], "antecedent"),
            ([Span(start=3843, end=3915)], "antecedent"),
            ([Span(start=3916, end=6667)], "antecedent"),
            ([Span(start=6668, end=9094)], "conclusion"),
        ],
    ),
    ("eds/clean/cas2", [([Span(start=0, end=6671)], "head")]),
    (
        "eds/clean/cas3",
        [
            ([Span(start=0, end=314)], "head"),
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


@pytest.mark.parametrize(("filepath", "expected_sections"), TEST_CONFIG)
def test_run(filepath, expected_sections):
    clean_text_segment = _get_clean_text_segment(filepath)

    section_tokenizer = SectionTokenizer()
    sections = section_tokenizer.run([clean_text_segment])

    assert len(sections) == len(expected_sections)
    for i, (spans, attr_value) in enumerate(expected_sections):
        assert sections[i].spans == spans
        assert sections[i].metadata["name"] == attr_value
        assert sections[i].attrs.get(label=section_tokenizer.output_label)[0].value == attr_value


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
    section_tokenizer = SectionTokenizer(section_dict=section_dict, section_rules=section_rules)
    sections = section_tokenizer.run([clean_text_segment])

    assert len(sections) == 2
    sections_antecedent = [section for section in sections if section.metadata["name"] == "antecedent_before_exam"]
    assert len(sections_antecedent) == 1
    section_examen = [section for section in sections if section.metadata["name"] == "exam_after_antecedent"]
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
    prov_1 = prov_tracer.get_prov(section_1.uid)
    assert prov_1.data_item == section_1
    assert prov_1.op_desc == tokenizer.description
    assert prov_1.source_data_items == [clean_text_segment]

    section_2 = sections[1]
    prov_2 = prov_tracer.get_prov(section_2.uid)
    assert prov_2.data_item == section_2
    assert prov_2.op_desc == tokenizer.description
    assert prov_2.source_data_items == [clean_text_segment]


def test_section_def_file_encoding_error():
    with pytest.raises(UnicodeError):
        SectionTokenizer.load_section_definition(filepath=_PATH_TO_DEFAULT_RULES, encoding="utf-16")


def test_section_def_file(tmp_path):
    filepath = tmp_path.joinpath("section.yml")
    section_dict = {
        "patient": ["SUBJECTIF"],
        "traitement": ["MÉDICAMENTS", "PLAN"],
        "allergies": ["ALLERGIES"],
        "examen clinique": ["EXAMEN PHYSIQUE"],
        "diagnostique": ["EVALUATION"],
    }
    treatment_rules = (
        SectionModificationRule(
            section_name="traitement",
            new_section_name="traitement_entree",
            other_sections=["diagnostique"],
            order="BEFORE",
        ),
        SectionModificationRule(
            section_name="traitement",
            new_section_name="traitement_sortie",
            other_sections=["diagnostique"],
            order="AFTER",
        ),
    )
    SectionTokenizer.save_section_definition(
        section_dict=section_dict,
        section_rules=treatment_rules,
        filepath=filepath,
        encoding="utf-8",
    )
    loaded_dict, loaded_rules = SectionTokenizer.load_section_definition(filepath, encoding="utf-8")
    assert loaded_dict == section_dict
    assert loaded_rules == treatment_rules
