import re

import pytest

pytest.importorskip(modname="torch", reason="torch is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher

_SPAN_OFFSET = 10


# mock of TokenClassificationPipeline class used by HFEntityMatcher
class _MockedPipeline:
    def __init__(self, *args, **kwargs):
        self._regexps_and_labels = [
            (re.compile(r"\basthma\b"), "problem"),
            (re.compile(r"\bdiabetes\b"), "problem"),
            (re.compile(r"\bventoline\b"), "treatment"),
        ]

    def __call__(self, texts):
        all_match_dicts = []
        for text in texts:
            match_dicts = [
                {
                    "start": match.start(),
                    "end": match.end(),
                    "entity_group": label,
                    "score": 1.0,
                }
                for regexp, label in self._regexps_and_labels
                for match in regexp.finditer(text)
            ]
            all_match_dicts.append(match_dicts)

        return all_match_dicts


@pytest.fixture(scope="module", autouse=True)
def _mocked_pipeline(module_mocker):
    module_mocker.patch(
        "medkit.text.ner.hf_entity_matcher.hf_utils.check_model_for_task_HF",
        return_value=True,
    )
    module_mocker.patch("transformers.pipeline", _MockedPipeline)


def _get_sentence_segment(text):
    return Segment(
        label="sentence",
        spans=[Span(_SPAN_OFFSET, _SPAN_OFFSET + len(text))],
        text=text,
    )


def test_single_match():
    """Basic behavior, single match in one input segment"""
    sentence = _get_sentence_segment("The patient has asthma.")
    matcher = HFEntityMatcher(model="mock-model")
    entities = matcher.run([sentence])
    assert len(entities) == 1

    # entity
    entity = entities[0]
    assert entity.label == "problem"
    assert entity.text == "asthma"
    assert entity.spans == [Span(26, 32)]

    # score attribute
    attrs = entity.attrs.get(label="score")
    assert len(attrs) == 1
    attr = attrs[0]
    assert attr.label == "score"
    assert 0.0 <= attr.value <= 1.0


def test_multiple_matches():
    """Basic behavior, multiple matches in multiple input segments"""
    sentence_1 = _get_sentence_segment("The patient has asthma and is using ventoline.")
    sentence_2 = _get_sentence_segment("The patient has diabetes.")
    sentences = [sentence_1, sentence_2]
    matcher = HFEntityMatcher(model="mock-model")
    entities = matcher.run(sentences)
    assert len(entities) == 3

    # 1st entity
    entity_1 = entities[0]
    assert entity_1.label == "problem"
    assert entity_1.text == "asthma"
    assert entity_1.spans == [Span(26, 32)]

    # 2nd entity
    entity_2 = entities[1]
    assert entity_2.label == "treatment"
    assert entity_2.text == "ventoline"
    assert entity_2.spans == [Span(46, 55)]

    # 3rd entity
    entity_3 = entities[2]
    assert entity_3.label == "problem"
    assert entity_3.text == "diabetes"
    assert entity_3.spans == [Span(26, 34)]


def test_attrs_to_copy():
    """Copying of selected attributes from input segment to created entity"""
    sentence = _get_sentence_segment("The patient has asthma.")
    # copied attribute
    neg_attr = Attribute(label="negation", value=False)
    sentence.attrs.add(neg_attr)
    # uncopied attribute
    sentence.attrs.add(Attribute(label="hypothesis", value=False))

    matcher = HFEntityMatcher(model="mock-model", attrs_to_copy=["negation"])
    entity = matcher.run([sentence])[0]

    assert len(entity.attrs.get(label="score")) == 1
    # only negation attribute was copied
    neg_attrs = entity.attrs.get(label="negation")
    assert len(neg_attrs) == 1
    assert len(entity.attrs.get(label="hypothesis")) == 0

    # copied attribute has same value but new id
    copied_neg_attr = neg_attrs[0]
    assert copied_neg_attr.value == neg_attr.value
    assert copied_neg_attr.uid != neg_attr.uid


def test_prov():
    """Generated provenance nodes"""
    # use file containing voice signal
    sentence_1 = _get_sentence_segment("The patient has asthma and is using ventoline.")
    sentence_2 = _get_sentence_segment("The patient has diabetes.")
    sentences = [sentence_1, sentence_2]

    matcher = HFEntityMatcher(model="mock-model")
    prov_tracer = ProvTracer()
    matcher.set_prov_tracer(prov_tracer)
    entities = matcher.run(sentences)
    assert len(entities) == 3

    # data item uid and operation uid are correct
    entity_1 = entities[0]
    prov_1 = prov_tracer.get_prov(entity_1.uid)
    assert prov_1.data_item == entity_1
    assert prov_1.op_desc == matcher.description

    # 1st and 2nd entities have 1st sentence as source
    assert prov_1.source_data_items == [sentence_1]
    entity_2 = entities[1]
    prov_2 = prov_tracer.get_prov(entity_2.uid)
    assert prov_2.source_data_items == [sentence_1]

    # 3d entity has 2nd sentence as source
    entity_3 = entities[2]
    prov_3 = prov_tracer.get_prov(entity_3.uid)
    assert prov_3.source_data_items == [sentence_2]
