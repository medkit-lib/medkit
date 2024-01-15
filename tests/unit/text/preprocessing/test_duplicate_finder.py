from medkit.core import Collection, ProvTracer
from medkit.core.text import Span, TextDocument
from medkit.text.preprocessing import DuplicateFinder, DuplicationAttribute

LINES = [
    "Patient has been admitted for high-blood pressure",
    "Days since last visit: 10",
    "Has been scheduled for cholesterol test",
    "Patient is also reporting joint pain",
    "Patient was released today",
]


def _get_docs():
    text_0 = "\n".join([LINES[0], LINES[1], LINES[2]])
    doc_0 = TextDocument(text_0, metadata={"creation_date": "2022-02-27"})

    text_1 = "\n".join([LINES[0], LINES[2], LINES[3]])
    doc_1 = TextDocument(text_1, metadata={"creation_date": "2022-04-10"})
    text_2 = "\n".join([LINES[0], LINES[2], LINES[4]])
    doc_2 = TextDocument(text_2, metadata={"creation_date": "2022-09-29"})
    return [doc_0, doc_1, doc_2]


def test_basic():
    """Basic usage, return duplicate segments"""
    detector = DuplicateFinder(output_label="duplicate")
    docs = _get_docs()
    collection = Collection(text_docs=docs)
    detector.run([collection])

    # 1st doc has zero duplicates
    doc_1 = docs[0]
    dup_segs = doc_1.anns.get(label="duplicate")
    assert len(dup_segs) == 0

    # 2d doc has 2 duplicates
    doc_2 = docs[1]
    dup_segs = doc_2.anns.get(label="duplicate")
    assert len(dup_segs) == 2

    # 1st duplicate
    seg_1 = dup_segs[0]
    assert seg_1.text == LINES[0]
    # check spans
    assert seg_1.spans == [Span(0, len(LINES[0]))]
    # check source attr
    attrs = seg_1.attrs.get(label="is_duplicate")
    assert len(attrs) == 1
    attrs = attrs[0]
    assert isinstance(attrs, DuplicationAttribute)
    assert attrs.value is True
    assert attrs.source_doc_id == doc_1.uid
    assert attrs.source_spans == [Span(0, len(LINES[0]))]
    # 2d duplicate
    seg_2 = dup_segs[1]
    assert seg_2.text == LINES[2]

    # 3d doc also has same duplicates as 2d doc
    doc_3 = docs[2]
    dup_segs = doc_3.anns.get(label="duplicate")
    assert dup_segs[0].text == LINES[0]
    assert dup_segs[1].text == LINES[2]


def test_only_nondup():
    """Return non-duplicate segments"""
    detector = DuplicateFinder(output_label="nonduplicate", segments_to_output="nondup")
    docs = _get_docs()
    collection = Collection(text_docs=docs)
    detector.run([collection])

    # 1st doc is fully original
    doc_1 = docs[0]
    nondup_segs = doc_1.anns.get(label="nonduplicate")
    assert len(nondup_segs) == 1
    seg = nondup_segs[0]
    assert seg.text == doc_1.text
    assert seg.spans == [Span(0, len(seg.text))]
    attrs = seg.attrs.get(label="is_duplicate")
    assert len(attrs) == 1
    attrs = attrs[0]
    assert isinstance(attrs, DuplicationAttribute)
    assert attrs.value is False
    assert attrs.source_doc_id is None
    assert attrs.source_doc_date is None
    assert attrs.source_spans is None

    # 2d doc has last line as original
    doc_2 = docs[1]
    nondup_segs = doc_2.anns.get(label="nonduplicate")
    print([s.text for s in nondup_segs])
    print([s.spans for s in nondup_segs])
    assert len(nondup_segs) == 1
    seg = nondup_segs[0]
    assert seg.text == "\n" + LINES[3]

    # 3d doc also has last line as original
    doc_3 = docs[2]
    nondup_segs = doc_3.anns.get(label="nonduplicate")
    assert len(nondup_segs) == 1
    seg = nondup_segs[0]
    assert seg.text == "\n" + LINES[4]


def test_both():
    """Return duplicate and non-duplicate segments"""
    detector = DuplicateFinder(output_label="deduplicated", segments_to_output="both")
    docs = _get_docs()
    collection = Collection(text_docs=docs)
    detector.run([collection])

    # 1st doc is fully original, has only 1 segment
    doc_1 = docs[0]
    segs = doc_1.anns.get(label="deduplicated")
    assert len(segs) == 1
    assert segs[0].attrs.get(label="is_duplicate")[0].value is False

    # 2d doc has 2 duplicate segment and 1 non-dup
    doc_2 = docs[1]
    segs = doc_2.anns.get(label="deduplicated")
    assert len(segs) == 3
    assert segs[0].attrs.get(label="is_duplicate")[0].value is True
    assert segs[1].attrs.get(label="is_duplicate")[0].value is True
    assert segs[2].attrs.get(label="is_duplicate")[0].value is False

    # same for 3d doc
    doc_3 = docs[1]
    segs = doc_3.anns.get(label="deduplicated")
    assert len(segs) == 3
    assert segs[0].attrs.get(label="is_duplicate")[0].value is True
    assert segs[1].attrs.get(label="is_duplicate")[0].value is True
    assert segs[2].attrs.get(label="is_duplicate")[0].value is False


def test_char_fingerprint():
    """Use char fingerprint type"""
    detector = DuplicateFinder(output_label="duplicate", fingerprint_type="char", min_duplicate_length=10)
    docs = _get_docs()
    collection = Collection(text_docs=docs)
    detector.run([collection])

    # 1st doc has zero duplicates
    doc_1 = docs[0]
    dup_segs = doc_1.anns.get(label="duplicate")
    assert len(dup_segs) == 0

    # 2d doc has 2 duplicates
    doc_2 = docs[1]
    dup_segs = doc_2.anns.get(label="duplicate")
    assert len(dup_segs) == 2

    # 3d doc also has same duplicates as 2d doc
    doc_3 = docs[2]
    dup_segs = doc_3.anns.get(label="duplicate")
    assert len(dup_segs) == 2


def test_date():
    """Use date in metadata to order documents"""
    detector = DuplicateFinder(
        output_label="duplicate",
        date_metadata_key="creation_date",
    )
    docs = _get_docs()
    # pass docs in reverse order to detector
    collection = Collection(text_docs=reversed(docs))
    detector.run([collection])

    # we should have the same results as if docs had been passed in creation order
    # (1st/older doc is considered as "source" of duplicates in other docs)
    doc_1 = docs[0]
    segs = doc_1.anns.get(label="duplicate")
    assert len(segs) == 0

    doc_2 = docs[1]
    segs = doc_2.anns.get(label="duplicate")
    assert len(segs) == 2

    doc_3 = docs[2]
    segs = doc_3.anns.get(label="duplicate")
    assert len(segs) == 2

    # the date of the source doc should be added to the attributes
    doc_2 = docs[1]
    seg = doc_2.anns.get(label="duplicate")[0]
    attr = seg.attrs.get(label="is_duplicate")[0]
    assert attr.source_doc_id == doc_1.uid
    assert attr.source_doc_date == doc_1.metadata["creation_date"]


def test_prov():
    detector = DuplicateFinder(output_label="deduplicated", segments_to_output="both")
    prov_tracer = ProvTracer()
    detector.set_prov_tracer(prov_tracer)

    docs = _get_docs()
    collection = Collection(text_docs=docs)
    detector.run([collection])

    doc_1 = docs[0]
    doc_2 = docs[1]
    segs = doc_2.anns.get(label="deduplicated")

    # provenance for duplicate
    dup_seg = segs[0]
    prov = prov_tracer.get_prov(dup_seg.uid)
    assert prov.data_item == dup_seg
    assert prov.op_desc == detector.description
    assert prov.source_data_items == [doc_1.raw_segment, doc_2.raw_segment]

    dup_attr = dup_seg.attrs.get(label="is_duplicate")[0]
    prov = prov_tracer.get_prov(dup_attr.uid)
    assert prov.data_item == dup_attr
    assert prov.op_desc == detector.description
    assert prov.source_data_items == [doc_1.raw_segment, doc_2.raw_segment]

    # provenance for non-duplicate
    nondup_seg = segs[2]
    prov = prov_tracer.get_prov(nondup_seg.uid)
    assert prov.data_item == nondup_seg
    assert prov.op_desc == detector.description
    assert prov.source_data_items == [doc_2.raw_segment]

    nondup_attr = nondup_seg.attrs.get(label="is_duplicate")[0]
    prov = prov_tracer.get_prov(nondup_attr.uid)
    assert prov.data_item == nondup_attr
    assert prov.op_desc == detector.description
    assert prov.source_data_items == [doc_2.raw_segment]


def test_dict():
    docs = _get_docs()
    collection = Collection(text_docs=docs)
    detector = DuplicateFinder(output_label="deduplicated", segments_to_output="both")
    detector.run([collection])

    seg_attr = docs[1].anns.get(label="deduplicated")[0].attrs.get(label="is_duplicate")[0]
    assert seg_attr == DuplicationAttribute.from_dict(seg_attr.to_dict())
