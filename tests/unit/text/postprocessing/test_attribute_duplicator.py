import pytest
from intervaltree import IntervalTree

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Segment, TextDocument, span_utils
from medkit.text.postprocessing import AttributeDuplicator, compute_nested_segments
from medkit.text.postprocessing.alignment_utils import _create_segments_tree


def _extract_segment(segment, ranges, label, uid=None):
    text, spans = span_utils.extract(segment.text, segment.spans, ranges)
    return Segment(label=label, spans=spans, text=text, uid=uid)


@pytest.fixture()
def doc():
    doc = TextDocument("""Sa mère présente douleurs abdominales mais le patient n'a pas une maladie.""")
    sentence = _extract_segment(doc.raw_segment, [(0, 73)], "sentence")
    syntagme_0 = _extract_segment(sentence, [(0, 37)], "syntagme", uid="syntagme_0")
    syntagme_1 = _extract_segment(sentence, [(37, 73)], "syntagme", uid="syntagme_1")
    # add is_family in sentence
    sentence.attrs.add(Attribute(label="is_family", value=True))
    sentence.attrs.add(Attribute(label="family_trigger", value="Mother"))

    # add is_negated in syntagmes
    syntagme_0.attrs.add(Attribute(label="is_negated", value=False))
    syntagme_1.attrs.add(Attribute(label="is_negated", value=True))

    # create entities (target)
    target_0 = _extract_segment(doc.raw_segment, [(17, 37)], "disease", uid="target_0")
    target_1 = _extract_segment(doc.raw_segment, [(66, 73)], "disease", uid="target_1")

    for ann in [sentence, syntagme_0, syntagme_1, target_0, target_1]:
        doc.anns.add(ann)
    return doc


def test_compute_nested_segments(doc):
    # align syntagme with entities
    source = doc.anns.get(label="syntagme")
    targets = doc.anns.get(label="disease")
    nested = compute_nested_segments(source_segments=source, target_segments=targets)
    assert len(nested) == 2
    assert len(nested[0][1]) == 1
    assert nested[0][0].uid == "syntagme_0"
    assert nested[0][1][0].uid == "target_0"
    assert nested[1][0].uid == "syntagme_1"
    assert nested[1][1][0].uid == "target_1"


def test_compute_nested_segments_target_no_fully_contained(doc):
    # align syntagme with entities
    # sytagme_0 goes from 0:37, target goes from 35:40
    # so, target should not be a nested segment
    source = [doc.anns.get_by_id("syntagme_0")]
    target = [_extract_segment(doc.raw_segment, [(35, 40)], "disease", uid="target_x")]

    nested = compute_nested_segments(source_segments=source, target_segments=target)
    assert len(nested) == 1
    assert len(nested[0][1]) == 0
    assert nested[0][0].uid == "syntagme_0"


def test__create_segments_tree(doc):
    targets = doc.anns.get(label="disease")
    tree = _create_segments_tree(target_segments=targets)
    assert isinstance(tree, IntervalTree)
    assert len(tree.overlap(17, 37)) == 1
    assert len(tree.overlap(63, 73)) == 1
    assert len(tree.overlap(0, 100)) == 2


def test_default_attribute_duplicator(doc):
    sentences = doc.anns.get(label="sentence")
    syntagmes = doc.anns.get(label="syntagme")
    targets = doc.anns.get(label="disease")

    # check no attrs in targets
    assert all(len(target.attrs) == 0 for target in targets)

    # define attr duplicators
    duplicator_1 = AttributeDuplicator(attr_labels=["is_family"])
    duplicator_2 = AttributeDuplicator(attr_labels=["is_negated"])

    # is_family was 'detected' in sentences
    duplicator_1.run(sentences, targets)
    # is_negated was 'detected' in syntagmes
    duplicator_2.run(syntagmes, targets)

    # check new attrs
    assert all(len(target.attrs) == 2 for target in targets)

    # check first target
    target = targets[0]
    negated = target.attrs.get(label="is_negated")[0]
    family = target.attrs.get(label="is_family")[0]
    assert len(target.attrs.get(label="family_trigger")) == 0
    assert not negated.value
    assert family.value

    # check second target
    target = targets[1]
    negated = target.attrs.get(label="is_negated")[0]
    family = target.attrs.get(label="is_family")[0]
    assert len(target.attrs.get(label="family_trigger")) == 0
    assert negated.value
    assert family.value


def test_duplicate_a_list_of_attrs_labels(doc):
    sentences = doc.anns.get(label="sentence")
    targets = doc.anns.get(label="disease")

    # check no attrs in targets
    assert all(len(target.attrs) == 0 for target in targets)

    # define attr duplicator
    propagator = AttributeDuplicator(attr_labels=["is_family", "family_trigger"])
    # attrs were 'detected' in sentences
    propagator.run(sentences, targets)

    # check new attrs
    attr_src_family = sentences[0].attrs.get(label="is_family")[0]
    attr_src_family_trigger = sentences[0].attrs.get(label="family_trigger")[0]

    assert all(len(target.attrs) == 2 for target in targets)
    for target in targets:
        assert len(target.attrs.get(label="is_negated")) == 0
        family = target.attrs.get(label="is_family")[0]
        family_trigger = target.attrs.get(label="family_trigger")[0]
        assert family.value == attr_src_family.value
        assert family_trigger.value == attr_src_family_trigger.value == "Mother"
        assert family.uid != attr_src_family.uid
        assert family_trigger.uid != attr_src_family_trigger.uid


def test_provenance(doc):
    sentences = doc.anns.get(label="sentence")
    targets = doc.anns.get(label="disease")

    prov_tracer = ProvTracer()
    duplicator_1 = AttributeDuplicator(attr_labels=["is_family"])
    duplicator_1.set_prov_tracer(prov_tracer)

    # is_family was 'detected' in sentences
    duplicator_1.run(sentences, targets)

    sentence_attr_1 = sentences[0].attrs.get(label="is_family")[0]
    attr_1 = targets[0].attrs.get(label="is_family")[0]

    attr_1_prov = prov_tracer.get_prov(attr_1.uid)
    assert attr_1_prov.data_item == attr_1
    assert attr_1_prov.op_desc == duplicator_1.description
    assert attr_1_prov.source_data_items == [sentence_attr_1]
