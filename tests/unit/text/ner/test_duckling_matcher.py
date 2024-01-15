import pytest

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.ner.duckling_matcher import DucklingMatcher

_TEXT = "The patient was admitted on 01/02/2001 and stayed for 3 days before leaving"
_OUTPUT_LABEL = "duckling"

_TIME_VALUE = {
    "grain": "day",
    "type": "value",
    "value": "2001-01-02T00:00:00.000-08:00",
    "values": [{"grain": "day", "type": "value", "value": "2001-01-02T00:00:00.000-08:00"}],
}

_DURATION_VALUE = {
    "value": 3,
    "day": 3,
    "type": "value",
    "unit": "day",
    "normalized": {"value": 259200, "unit": "second"},
}

_TIME_RESPONSE_DATA = {
    "dim": "time",
    "body": "on 01/02/2001",
    "start": 25,
    "end": 38,
    "value": _TIME_VALUE,
}

_DURATION_RESPONSE_DATA = {
    "dim": "duration",
    "body": "3 days",
    "start": 54,
    "end": 60,
    "value": _DURATION_VALUE,
}


class _MockHTTPResponse:
    def __init__(self, data):
        self.status_code = 200
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def _mock_requests_get(url, timeout):
    return _MockHTTPResponse(None)


def _mock_requests_post(url, data, timeout):
    if "dims" not in data:
        response_data = [_TIME_RESPONSE_DATA, _DURATION_RESPONSE_DATA]
    else:
        response_data = []
        if "time" in data["dims"]:
            response_data.append(_TIME_RESPONSE_DATA)
        if "duration" in data["dims"]:
            response_data.append(_DURATION_RESPONSE_DATA)
    return _MockHTTPResponse(response_data)


@pytest.fixture(scope="module", autouse=True)
def _mocked_requests(module_mocker):
    module_mocker.patch("requests.get", _mock_requests_get)
    module_mocker.patch("requests.post", _mock_requests_post)


def _get_sentence_segment(text=_TEXT):
    return Segment(
        label="sentence",
        spans=[Span(0, len(text))],
        text=text,
    )


def test_single_dim():
    sentence = _get_sentence_segment()

    matcher = DucklingMatcher(
        output_label=_OUTPUT_LABEL,
        version="MOCK",
        locale="en",
        dims=["time"],
    )
    entities = matcher.run([sentence])

    # entity
    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "time"
    assert entity.text == "on 01/02/2001"
    assert entity.spans == [Span(25, 38)]

    # normalization attribute
    attrs = entity.attrs.get(label=_OUTPUT_LABEL)
    assert len(attrs) == 1
    attr = attrs[0]
    assert attr.label == _OUTPUT_LABEL
    assert attr.value == _TIME_VALUE
    assert attr.metadata["version"] == "MOCK"


def test_multiple_dims():
    sentence = _get_sentence_segment()

    matcher = DucklingMatcher(
        output_label=_OUTPUT_LABEL,
        version="MOCK",
        locale="en",
        dims=["time", "duration"],
    )
    entities = matcher.run([sentence])
    assert len(entities) == 2

    # 1st entity (time)
    entity_1 = entities[0]
    assert entity_1.label == "time"
    assert entity_1.text == "on 01/02/2001"
    assert entity_1.spans == [Span(25, 38)]

    attr_1 = entity_1.attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.label == "duckling"
    assert attr_1.value == _TIME_VALUE

    # 2d entity (duration)
    entity_2 = entities[1]
    assert entity_2.label == "duration"
    assert entity_2.text == "3 days"
    assert entity_2.spans == [Span(54, 60)]

    attr_2 = entity_2.attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.label == "duckling"
    assert attr_2.value == _DURATION_VALUE


def test_all_dims():
    sentence = _get_sentence_segment()

    matcher = DucklingMatcher(
        output_label=_OUTPUT_LABEL,
        version="MOCK",
        locale="en",
    )
    entities = matcher.run([sentence])
    assert len(entities) == 2


def test_attrs_to_copy():
    sentence = _get_sentence_segment()
    # copied attribute
    neg_attr = Attribute(label="negation", value=True)
    sentence.attrs.add(neg_attr)
    # uncopied attribute
    sentence.attrs.add(Attribute(label="hypothesis", value=True))

    matcher = DucklingMatcher(
        output_label=_OUTPUT_LABEL,
        version="MOCK",
        locale="en",
        dims=["time"],
        attrs_to_copy=["negation"],
    )
    entity = matcher.run([sentence])[0]

    assert len(entity.attrs.get(label=_OUTPUT_LABEL)) == 1
    # only negation attribute was copied
    neg_attrs = entity.attrs.get(label="negation")
    assert len(neg_attrs) == 1
    assert len(entity.attrs.get(label="hypothesis")) == 0

    # copied attribute has same value but new id
    copied_neg_attr = neg_attrs[0]
    assert copied_neg_attr.value == neg_attr.value
    assert copied_neg_attr.uid != neg_attr.uid


def test_prov():
    sentence = _get_sentence_segment()

    matcher = DucklingMatcher(
        output_label=_OUTPUT_LABEL,
        version="MOCK",
        locale="en",
        dims=["time"],
    )
    prov_tracer = ProvTracer()
    matcher.set_prov_tracer(prov_tracer)

    entity = matcher.run([sentence])[0]

    entity_prov = prov_tracer.get_prov(entity.uid)
    assert entity_prov.data_item == entity
    assert entity_prov.op_desc == matcher.description
    assert entity_prov.source_data_items == [sentence]

    attr = entity.attrs.get(label=_OUTPUT_LABEL)[0]
    attr_prov = prov_tracer.get_prov(attr.uid)
    assert attr_prov.data_item == attr
    assert attr_prov.op_desc == matcher.description
    assert attr_prov.source_data_items == [sentence]
