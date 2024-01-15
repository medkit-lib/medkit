import pytest

from medkit.core.attribute import Attribute
from medkit.core.prov_tracer import ProvTracer
from medkit.core.text import Segment, Span

pytest.importorskip(modname="nlstruct", reason="nlstruct is not installed")
pytest.importorskip(modname="torch", reason="torch is not installed")
pytest.importorskip(modname="huggingface_hub", reason="huggingface-hub is not installed")


from medkit.text.ner.nlstruct_entity_matcher import NLStructEntityMatcher


# mock of NLStructModel class
class _MockedNLStructModel:
    def __init__(self):
        pass

    def eval(self):
        pass

    def predict(self, doc):
        doc_id = (doc["doc_id"],)
        text = doc["text"]
        return {
            "doc_id": doc_id,
            "text": text,
            "entities": [
                {
                    "entity_id": 0,
                    "label": ["MISC"],
                    "attributes": [],
                    "fragments": [
                        {
                            "begin": 0,
                            "end": len(text),
                            "label": "MISC",
                            "text": text,
                        }
                    ],
                    "confidence": 0.0,
                }
            ],
        }


@pytest.fixture(scope="module", autouse=True)
def _mocked_nlstruct_modules(module_mocker):
    module_mocker.patch(
        "medkit.text.ner.nlstruct_entity_matcher.NLStructEntityMatcher._load_from_checkpoint_dir",
        return_value=_MockedNLStructModel(),
    )
    module_mocker.patch(
        "medkit.text.ner.nlstruct_entity_matcher.huggingface_hub.snapshot_download",
        return_value=".",
    )


def _get_segment(text):
    return Segment(text=text, spans=[Span(0, len(text))], label="segment")


def test_single_match():
    text = "The patient has asthma."
    segment = _get_segment(text)
    matcher = NLStructEntityMatcher(model_name_or_dirpath="mock-model")
    entities = matcher.run([segment])

    assert len(entities) == 1

    # entity
    entity = entities[0]
    assert entity.label == "MISC"
    assert entity.text == text
    assert entity.spans == [Span(0, len(text))]

    # score attribute
    attrs = entity.attrs.get(label="confidence")
    assert len(attrs) == 1
    attr = attrs[0]
    assert attr.value == 0.0


def test_attrs_to_copy():
    """Copying of selected attributes from input segment to created entity"""
    sentence = _get_segment("The patient has asthma.")
    # copied attribute
    neg_attr = Attribute(label="negation", value=False)
    sentence.attrs.add(neg_attr)
    # uncopied attribute
    sentence.attrs.add(Attribute(label="hypothesis", value=False))

    matcher = NLStructEntityMatcher(model_name_or_dirpath="mock-model", attrs_to_copy=["negation"])
    entity = matcher.run([sentence])[0]

    assert len(entity.attrs.get(label="confidence")) == 1
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
    sentence = _get_segment("The patient has asthma.")
    sentences = [sentence]

    matcher = NLStructEntityMatcher(model_name_or_dirpath="mock-model")
    prov_tracer = ProvTracer()
    matcher.set_prov_tracer(prov_tracer)
    entities = matcher.run(sentences)
    assert len(entities) == 1

    # check prov entity
    entity_1 = entities[0]
    prov_1 = prov_tracer.get_prov(entity_1.uid)
    assert prov_1.data_item == entity_1
    assert prov_1.op_desc == matcher.description
    assert prov_1.source_data_items == [sentence]

    # check prov attr
    attr = entity_1.attrs.get(label="confidence")[0]
    prov_2 = prov_tracer.get_prov(attr.uid)
    assert prov_2.data_item == attr
    assert prov_2.op_desc == matcher.description
    assert prov_2.source_data_items == [sentence]
