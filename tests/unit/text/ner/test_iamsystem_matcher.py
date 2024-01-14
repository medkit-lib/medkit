from iamsystem import Entity as Term
from iamsystem import Matcher

from medkit.core import Attribute
from medkit.core.text import ModifiedSpan, Segment, Span
from medkit.text.ner import IAMSystemMatcher
from medkit.text.ner.iamsystem_matcher import MedkitKeyword


def test_simple_matcher():
    labels = ["acute respiratory distress syndrome", "diarrrhea"]
    term = Term(label=labels[0], kb_id="test")
    matcher = Matcher.build(keywords=[*labels, term])
    text = "Pt c/o Acute Respiratory Distress Syndrome and diarrrhea"

    medkit_matcher = IAMSystemMatcher(matcher=matcher)
    segment = Segment(label="raw_text", text=text, spans=[Span(0, len(text))])
    entities = medkit_matcher.run([segment])
    for entity in entities:
        assert len(entity.text) == sum(sp.length for sp in entity.spans)
    assert entities[0].label == labels[0]
    assert entities[0].attrs.get_norms()[0].kb_id == "test"
    assert entities[1].attrs.get_norms() == []
    assert entities[1].label == labels[1]


def _get_first_entity(matcher: Matcher):
    """Utility function to avoid repeating these lines."""
    text = "calcium blood level"
    medkit_matcher = IAMSystemMatcher(matcher=matcher)
    segment = Segment(label="raw_text", text=text, spans=[Span(0, len(text))])
    return medkit_matcher.run([segment])[0]


def test_matcher_window():
    matcher = Matcher.build(keywords=["calcium level"], w=2)
    entity = _get_first_entity(matcher)
    assert entity.label == "calcium level"
    assert entity.spans == [
        Span(0, 7),
        ModifiedSpan(1, replaced_spans=[]),
        Span(14, 19),
    ]


def test_matcher_with_keyword_ent_label():
    """Check entity label is the keyword's label for a Keyword class."""
    from iamsystem import Keyword

    matcher = Matcher.build(keywords=[Keyword(label="CALCIUM")])
    entity = _get_first_entity(matcher)
    assert entity.label == "CALCIUM"


class UserKeyword:
    def __init__(self, label: str, kb_id: str, kb_name, ent_label):
        self.label = label
        self.kb_id = kb_id
        self.kb_name = kb_name
        self.ent_label = ent_label


def test_matcher_userkw_ent_label():
    """Check ent_label is set, entity is the label."""
    matcher = Matcher.build(
        keywords=[
            UserKeyword(
                label="calcium",
                kb_id="LOINC-2",
                kb_name="UMLS",
                ent_label="biological measurement",
            )
        ]
    )
    entity = _get_first_entity(matcher)
    assert entity.label == "biological measurement"


def test_matcher_userkw_multiple_ent_label():
    """When multiple ent_labels, the entity label is the first keyword's
    ent_label (if set).
    """
    matcher = Matcher.build(
        keywords=[
            UserKeyword(label="calcium", kb_id="LOINC-2", kb_name="UMLS", ent_label="first one"),
            UserKeyword(
                label="calcium",
                kb_id="LOINC-2",
                kb_name="UMLS",
                ent_label="biological measurement",
            ),
        ]
    )
    entity = _get_first_entity(matcher)
    assert entity.label == "first one"


def test_matcher_userkw_multiple_ent_label_first_none():
    """When multiple ent_labels, the entity label is the first keyword's
    ent_label that is not None.
    """
    matcher = Matcher.build(
        keywords=[
            UserKeyword(label="calcium", kb_id="LOINC-2", kb_name="UMLS", ent_label=None),
            UserKeyword(label="calcium", kb_id="LOINC-2", kb_name="UMLS", ent_label="first one"),
            UserKeyword(
                label="calcium",
                kb_id="LOINC-2",
                kb_name="UMLS",
                ent_label="biological measurement",
            ),
        ]
    )
    entity = _get_first_entity(matcher)
    assert entity.label == "first one"


def test_matcher_userkw_ent_label_none():
    """When ent_label is None, it should return the keyword's label."""
    matcher = Matcher.build(keywords=[UserKeyword(label="calcium", kb_id="LOINC-2", kb_name="UMLS", ent_label=None)])
    entity = _get_first_entity(matcher)
    assert entity.label == "calcium"


def test_matcher_userkw_len_norm_attr():
    """When one iamsystem's annotation has 4 keywords (e.g. Keywords have the
    same label), IAMSystemMatcher create 3 norm attributes
    because first one do not have kb_id or kb_name values.
    """
    matcher = Matcher.build(
        keywords=[
            "calcium",
            Term(label="calcium", kb_id="LOINC-1"),
            UserKeyword(label="calcium", kb_id="LOINC-2", kb_name="UMLS", ent_label=None),
            UserKeyword(label="calcium", kb_id="LOINC-2", kb_name="UMLS", ent_label=None),
        ]
    )
    entity = _get_first_entity(matcher)
    assert len(entity.attrs.get_norms()) == 3


def test_matcher_kb_name_kb_id():
    """Check kb_name and kb_id are correctly set in norm_attributes."""
    from iamsystem import Keyword

    matcher = Matcher.build(
        keywords=[
            UserKeyword(label="calcium", kb_id="LOINC-2", kb_name="UMLS", ent_label=None),
            MedkitKeyword(label="calcium", kb_id=None, kb_name="WIKIPEDIA", ent_label=None),
            Keyword(label="calcium"),
        ]
    )
    entity = _get_first_entity(matcher)
    assert len(entity.attrs.get_norms()) == 2
    assert entity.attrs.get_norms()[0].kb_name == "UMLS"
    assert entity.attrs.get_norms()[0].kb_id == "LOINC-2"
    assert entity.attrs.get_norms()[1].kb_name == "WIKIPEDIA"
    assert entity.attrs.get_norms()[1].kb_id is None


def provide_umls_label(keywords):
    """Return first UMLS ent label or None."""
    for kw in keywords:
        if kw.kb_name == "UMLS":
            return kw.ent_label
    return None


def test_matcher_custom_label_provider():
    """Test overriding default label_provider."""
    matcher = Matcher.build(
        keywords=[
            MedkitKeyword(label="calcium", kb_id=None, kb_name="WIKIPEDIA", ent_label="compound"),
            MedkitKeyword(
                label="calcium",
                kb_id=None,
                kb_name="UMLS",
                ent_label="biological measurement",
            ),
        ],
    )
    text = "calcium blood level"
    medkit_matcher = IAMSystemMatcher(matcher=matcher, label_provider=provide_umls_label)
    segment = Segment(label="raw_text", text=text, spans=[Span(0, len(text))])
    entity = medkit_matcher.run([segment])[0]
    assert entity.label == "biological measurement"


def test_attrs_to_copy():
    """Copying of selected attributes from input segment to created entity"""
    text = "The patient has asthma"
    sentence = Segment(label="sentence", text=text, spans=[Span(0, len(text))])
    # copied attribute
    neg_attr = Attribute(label="negation", value=False)
    sentence.attrs.add(neg_attr)
    # uncopied attribute
    sentence.attrs.add(Attribute(label="hypothesis", value=False))

    # use iamsystem matcher
    matcher = Matcher.build(keywords=["asthma"])
    matcher = IAMSystemMatcher(matcher=matcher, attrs_to_copy=["negation"])
    entity = matcher.run([sentence])[0]

    # only negation attribute was copied
    neg_attrs = entity.attrs.get(label="negation")
    assert len(neg_attrs) == 1
    assert len(entity.attrs.get(label="hypothesis")) == 0

    # copied attribute has same value but new id
    copied_neg_attr = neg_attrs[0]
    assert copied_neg_attr.value == neg_attr.value
    assert copied_neg_attr.uid != neg_attr.uid
