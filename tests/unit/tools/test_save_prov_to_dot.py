from medkit.core import Attribute, OperationDescription, ProvTracer, generate_id
from medkit.core.text import Entity, Segment, Span
from medkit.tools import save_prov_to_dot


def _get_segment_and_entity(with_attr=False):
    sentence_segment = Segment(label="sentence", text="This is a sentence.", spans=[Span(0, 19)])
    syntagma_segment = Segment(label="syntagma", text="a sentence", spans=[Span(8, 18)])
    entity = Entity(label="word", spans=[Span(10, 18)], text="sentence")
    if with_attr:
        attr = Attribute(label="negated", value=False)
        entity.attrs.add(attr)

    return sentence_segment, syntagma_segment, entity


def _build_prov(prov_tracer, sentence_segment, syntagma_segment, entity):
    tokenizer_desc = OperationDescription(uid=generate_id(), name="SyntagmaTokenizer")
    prov_tracer.add_prov(syntagma_segment, tokenizer_desc, source_data_items=[sentence_segment])

    matcher_desc = OperationDescription(uid=generate_id(), name="EntityMatcher")
    prov_tracer.add_prov(entity, matcher_desc, source_data_items=[syntagma_segment])

    for attr in entity.attrs:
        # add attribute to entity
        neg_detector_desc = OperationDescription(uid=generate_id(), name="NegationDetector")
        prov_tracer.add_prov(attr, neg_detector_desc, source_data_items=[syntagma_segment])


def test_basic(tmp_path):
    """Basic usage"""
    # build provenance
    sentence_segment, syntagma_segment, entity = _get_segment_and_entity()
    prov_tracer = ProvTracer()
    _build_prov(prov_tracer, sentence_segment, syntagma_segment, entity)

    # export to dot
    dot_file = tmp_path / "prov.dot"
    save_prov_to_dot(prov_tracer, dot_file)
    dot_text = dot_file.read_text()

    # check dot entries
    assert f'"{sentence_segment.uid}" [label="sentence: This is a sentence."];\n' in dot_text
    assert f'"{syntagma_segment.uid}" [label="syntagma: a sentence"];\n' in dot_text
    assert f'"{entity.uid}" [label="word: sentence"];\n' in dot_text
    assert f'"{sentence_segment.uid}" -> "{syntagma_segment.uid}"' ' [label="SyntagmaTokenizer"];\n' in dot_text
    assert f'"{syntagma_segment.uid}" -> "{entity.uid}" [label="EntityMatcher"];\n' in dot_text


def test_custom_format(tmp_path):
    """Custom data item and operation formatters"""
    # build provenance
    sentence_segment, syntagma_segment, entity = _get_segment_and_entity()
    prov_tracer = ProvTracer()
    _build_prov(prov_tracer, sentence_segment, syntagma_segment, entity)

    dot_file = tmp_path / "prov.dot"
    save_prov_to_dot(
        prov_tracer,
        dot_file,
        # change formatting
        data_item_formatters={Segment: lambda s: s.text},
        op_formatter=lambda o: f"Operation: {o.name}",
    )
    dot_text = dot_file.read_text()

    # check dot entries
    # segments are formatted differently
    assert f'"{sentence_segment.uid}" [label="This is a sentence."];\n' in dot_text
    # operations are formatted differently
    assert (
        f'"{sentence_segment.uid}" -> "{syntagma_segment.uid}"' ' [label="Operation: SyntagmaTokenizer"];\n' in dot_text
    )


def test_attrs(tmp_path):
    """Display attribute links"""
    # build provenance
    sentence_segment, syntagma_segment, entity = _get_segment_and_entity(with_attr=True)
    prov_tracer = ProvTracer()
    _build_prov(prov_tracer, sentence_segment, syntagma_segment, entity)

    # export to dot
    dot_file = tmp_path / "prov.dot"
    save_prov_to_dot(prov_tracer, dot_file)
    dot_text = dot_file.read_text()

    # check attribute link in dot entries
    attr = entity.attrs.get()[0]
    assert f'"{entity.uid}" -> "{attr.uid}" [style=dashed, color=grey,' ' label="attr", fontcolor=grey];\n' in dot_text


def test_sub_prov(tmp_path):
    """Handling of nested provenance tracers"""
    prov_tracer = ProvTracer()

    # build provenance for inner graph
    sentence_segment, syntagma_segment, entity = _get_segment_and_entity()
    sub_prov_tracer = ProvTracer(store=prov_tracer.store)
    _build_prov(sub_prov_tracer, sentence_segment, syntagma_segment, entity)

    # wrap it in outer pipeline graph
    pipeline_desc = OperationDescription(uid=generate_id(), name="Pipeline")
    prov_tracer.add_prov_from_sub_tracer([entity], pipeline_desc, sub_prov_tracer)

    # render dot, not expanding sub provenance
    dot_file = tmp_path / "prov.dot"
    save_prov_to_dot(prov_tracer, dot_file, max_sub_prov_depth=0)
    dot_text = dot_file.read_text()

    # must have a dot entry for outer pipeline operation
    assert f'"{sentence_segment.uid}" -> "{entity.uid}" [label="Pipeline"];\n' in dot_text

    # render dot, expanding all sub provenance
    dot_file_full = tmp_path / "prov_full.dot"
    save_prov_to_dot(prov_tracer, dot_file_full, max_sub_prov_depth=None)
    dot_text_full = dot_file_full.read_text()

    # must have a dot entry for inner operations in sub provenance
    assert f'"{sentence_segment.uid}" -> "{syntagma_segment.uid}"' ' [label="SyntagmaTokenizer"];\n' in dot_text_full
    assert f'"{syntagma_segment.uid}" -> "{entity.uid}" [label="EntityMatcher"];\n' in dot_text_full
