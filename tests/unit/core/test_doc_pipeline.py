from __future__ import annotations

from medkit.core import (
    AnnotationContainer,
    Attribute,
    AttributeContainer,
    Pipeline,
    PipelineStep,
    generate_id,
)
from medkit.core.doc_pipeline import DocPipeline

_FULL_TEXT = (
    "This is a sentence. This is another sentence. This is the last sentence.\nThis is"
    " a sentence with a different label. This is another sentence with a different"
    " label."
)
_SENTENCES = [
    "This is a sentence",
    "This is another sentence",
    "This is the last sentence",
]
_ALT_SENTENCES = [
    "This is a sentence with a different label",
    "This is another sentence with a different label",
]
_ENTITIES = ["Entity1", "Entity2"]


class _TextAnnotation:
    """Mock text annotation"""

    def __init__(self, label, text):
        self.uid = generate_id()
        self.label = label
        self.text = text
        self.keys = set()
        self.attrs = AttributeContainer(owner_id=self.uid)


class _TextDocument:
    """Mock text document"""

    def __init__(self, text):
        self.uid = generate_id()
        self.anns = AnnotationContainer(doc_id=self.uid)
        self.raw_segment = _TextAnnotation(label="raw", text=text)


def _get_doc():
    doc = _TextDocument(text=_FULL_TEXT)
    for text in _SENTENCES:
        ann = _TextAnnotation(label="sentence", text=text)
        doc.anns.add(ann)
    for text in _ALT_SENTENCES:
        doc.anns.add(_TextAnnotation(label="alt_sentence", text=text))
    for text in _ENTITIES:
        doc.anns.add(_TextAnnotation(label="entity", text=text))
    return doc


class _Uppercaser:
    """Mock processing operation uppercasing annotations"""

    def __init__(self, output_label):
        self.id = generate_id()
        self.output_label = output_label

    def run(self, anns):
        uppercase_anns = []
        for ann in anns:
            uppercase_ann = _TextAnnotation(
                label=self.output_label,
                text=ann.text.upper(),
            )
            uppercase_anns.append(uppercase_ann)
        return uppercase_anns


class _Prefixer:
    """Mock processing operation prefixing annotations"""

    def __init__(self, output_label, prefix):
        self.id = generate_id()
        self.output_label = output_label
        self.prefix = prefix

    def run(self, anns):
        prefixed_anns = []
        for ann in anns:
            prefixed_ann = _TextAnnotation(
                label=self.output_label,
                text=self.prefix + ann.text,
            )
            prefixed_anns.append(prefixed_ann)
        return prefixed_anns


class _AttributeAdder:
    """Mock processing operation adding attributes to existing annotations"""

    def __init__(self, output_label):
        self.id = generate_id()
        self.output_label = output_label

    def run(self, anns):
        for ann in anns:
            ann.attrs.add(Attribute(label=self.output_label, value=True))


def test_single_step():
    """Minimalist doc pipeline with only one step, retrieving input annotations from doc"""
    uppercaser = _Uppercaser(output_label="uppercased_sentence")
    step = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )
    pipeline = Pipeline(
        steps=[step],
        input_keys=step.input_keys,
        output_keys=step.output_keys,
    )
    doc_pipeline = DocPipeline(pipeline=pipeline, labels_by_input_key={"SENTENCE": ["sentence"]})

    doc = _get_doc()
    doc_pipeline.run([doc])

    sentence_anns = doc.anns.get(label="sentence")

    # new annotations were added to the document
    uppercased_anns = doc.anns.get(label="uppercased_sentence")

    # operation was properly called to generate new annotations
    assert [a.text.upper() for a in sentence_anns] == [a.text for a in uppercased_anns]


def test_multiple_steps():
    """Simple pipeline doc with 2 consecutive steps"""
    uppercaser = _Uppercaser(output_label="uppercased_sentence")
    step_1 = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix = "Hello! "
    prefixer = _Prefixer(output_label="prefixed_uppercased_sentence", prefix=prefix)
    step_2 = PipelineStep(
        operation=prefixer,
        input_keys=["UPPERCASE"],
        output_keys=["PREFIX"],
    )

    pipeline = Pipeline(
        steps=[step_1, step_2],
        input_keys=step_1.input_keys,
        output_keys=step_2.output_keys,
    )

    doc_pipeline = DocPipeline(pipeline=pipeline, labels_by_input_key={"SENTENCE": ["sentence"]})
    doc = _get_doc()
    doc_pipeline.run([doc])

    sentence_anns = doc.anns.get(label="sentence")

    # new annotations were added to the document
    prefixed_uppercased_anns = doc.anns.get(label="prefixed_uppercased_sentence")
    assert len(prefixed_uppercased_anns) == len(sentence_anns)

    # operations were properly called and in the correct order to generate new annotations
    expected_texts = [prefix + a.text.upper() for a in sentence_anns]
    assert [a.text for a in prefixed_uppercased_anns] == expected_texts


def test_no_output():
    """Doc pipeline having no output, because it has an operation that
    modifies the annotations it receives by adding attributes to them
    """
    attribute_adder = _AttributeAdder(output_label="validated")
    step_1 = PipelineStep(
        operation=attribute_adder,
        input_keys=["SENTENCE"],
        output_keys=[],
    )

    pipeline = Pipeline(steps=[step_1], input_keys=step_1.input_keys, output_keys=[])

    doc_pipeline = DocPipeline(pipeline=pipeline, labels_by_input_key={"SENTENCE": ["sentence"]})
    doc = _get_doc()
    doc_pipeline.run([doc])

    sentence_anns = doc.anns.get(label="sentence")
    for ann in sentence_anns:
        attrs = ann.attrs.get(label="validated")
        assert len(attrs) == 1
        assert attrs[0].value is True


def test_multiple_outputs():
    """Doc pipeline with more than 1 output"""
    uppercaser = _Uppercaser(output_label="uppercased_sentence")
    step_1 = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix = "Hello! "
    prefixer = _Prefixer(output_label="prefixed_sentence", prefix=prefix)
    step_2 = PipelineStep(
        operation=prefixer,
        input_keys=["SENTENCE"],
        output_keys=["PREFIX"],
    )

    pipeline = Pipeline(
        steps=[step_1, step_2],
        input_keys=step_1.input_keys,
        output_keys=step_1.output_keys + step_2.output_keys,
    )

    doc_pipeline = DocPipeline(pipeline=pipeline, labels_by_input_key={"SENTENCE": ["sentence"]})
    doc = _get_doc()
    doc_pipeline.run([doc])

    sentence_anns = doc.anns.get(label="sentence")
    uppercased_anns = doc.anns.get(label="uppercased_sentence")
    prefixed_anns = doc.anns.get(label="prefixed_sentence")

    expected_texts = [a.text.upper() for a in sentence_anns]
    assert [a.text for a in uppercased_anns] == expected_texts

    expected_texts = [prefix + a.text for a in sentence_anns]
    assert [a.text for a in prefixed_anns] == expected_texts


def test_labels_for_input_key():
    """Doc pipeline with several label to input key associations,
    including 2 labels associated to the same key
    """
    doc = _get_doc()

    uppercaser = _Uppercaser(output_label="uppercased_sentence")
    step_1 = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix = "Hello! "
    prefixer = _Prefixer(output_label="prefixed_entity", prefix=prefix)
    step_2 = PipelineStep(
        operation=prefixer,
        input_keys=["ENTITY"],
        output_keys=["PREFIX"],
    )

    labels_by_input_key = {
        "SENTENCE": ["sentence", "alt_sentence"],
        "ENTITY": ["entity"],
    }

    pipeline = Pipeline(
        steps=[step_1, step_2],
        input_keys=step_1.input_keys + step_2.input_keys,
        output_keys=step_1.output_keys + step_2.output_keys,
    )

    doc_pipeline = DocPipeline(
        pipeline=pipeline,
        labels_by_input_key=labels_by_input_key,
    )
    doc_pipeline.run([doc])
    sentence_anns = doc.anns.get(label="sentence")
    alt_sentence_anns = doc.anns.get(label="alt_sentence")
    uppercased_sentence_anns = doc.anns.get(label="uppercased_sentence")
    assert len(uppercased_sentence_anns) == len(sentence_anns) + len(alt_sentence_anns)

    expected_texts = [a.text.upper() for a in (sentence_anns + alt_sentence_anns)]
    assert [a.text for a in uppercased_sentence_anns] == expected_texts

    entity_anns = doc.anns.get(label="entity")
    prefixed_entity_anns = doc.anns.get(label="prefixed_entity")
    assert len(prefixed_entity_anns) == len(entity_anns)

    expected_texts = [prefix + a.text for a in entity_anns]
    assert [a.text for a in prefixed_entity_anns] == expected_texts


def test_labels_for_input_key_different_order():
    """Doc pipeline with several labels to input key associations,
    provided in different order than the underlying pipeline's input keys
    (regression test)
    """
    doc = _get_doc()

    uppercaser = _Uppercaser(output_label="uppercased_sentence")
    step_1 = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix = "Hello! "
    prefixer = _Prefixer(output_label="prefixed_entity", prefix=prefix)
    step_2 = PipelineStep(
        operation=prefixer,
        input_keys=["ENTITY"],
        output_keys=["PREFIX"],
    )

    pipeline = Pipeline(
        steps=[step_1, step_2],
        input_keys=["SENTENCE", "ENTITY"],
        output_keys=["UPPERCASE", "PREFIX"],
    )

    # ordering of labels_by_input_key different that ordering of pipeline input_keys
    labels_by_input_key = {
        "ENTITY": ["entity"],
        "SENTENCE": ["sentence"],
    }

    doc_pipeline = DocPipeline(
        pipeline=pipeline,
        labels_by_input_key=labels_by_input_key,
    )
    doc_pipeline.run([doc])

    sentence_anns = doc.anns.get(label="sentence")
    uppercased_sentence_anns = doc.anns.get(label="uppercased_sentence")
    expected_texts = [a.text.upper() for a in (sentence_anns)]
    assert [a.text for a in uppercased_sentence_anns] == expected_texts

    entity_anns = doc.anns.get(label="entity")
    prefixed_entity_anns = doc.anns.get(label="prefixed_entity")
    expected_texts = [prefix + a.text for a in entity_anns]
    assert [a.text for a in prefixed_entity_anns] == expected_texts


def test_no_labels_for_input_key():
    """DocPipeline defaulting to raw_segment for the unique input_key of the underlying pipeline"""
    uppercaser = _Uppercaser(output_label="uppercased_full_text")
    step = PipelineStep(
        operation=uppercaser,
        input_keys=["FULL_TEXT"],
        output_keys=["UPPERCASE"],
    )
    pipeline = Pipeline(
        steps=[step],
        input_keys=["FULL_TEXT"],
        output_keys=["UPPERCASE"],
    )
    doc_pipeline = DocPipeline(pipeline=pipeline)

    doc = _get_doc()
    doc_pipeline.run([doc])

    # new annotation was added to the document
    uppercased_ann = doc.anns.get(label="uppercased_full_text")[0]
    # operation was properly called to generate new annotation
    assert uppercased_ann.text == doc.raw_segment.text.upper()


def test_nested_pipeline():
    """DocPipeline wrapped in a Pipeline"""
    # build inner pipeline
    uppercaser = _Uppercaser(output_label="uppercased_sentence")
    sub_step = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    sub_pipeline = Pipeline(
        steps=[sub_step],
        input_keys=sub_step.input_keys,
        output_keys=sub_step.output_keys,
    )
    doc_sub_pipeline = DocPipeline(pipeline=sub_pipeline, labels_by_input_key={"SENTENCE": ["sentence"]})

    # wrap it in main pipeline
    step = PipelineStep(
        operation=doc_sub_pipeline,
        input_keys=["DOC"],
        output_keys=[],
    )

    pipeline = Pipeline(steps=[step], input_keys=["DOC"], output_keys=[])

    doc = _get_doc()
    pipeline.run([doc])
    sentence_anns = doc.anns.get(label="sentence")
    # new annotations were added to the document
    uppercased_anns = doc.anns.get(label="uppercased_sentence")
    # operation was properly called to generate new annotations
    assert [a.text.upper() for a in sentence_anns] == [a.text for a in uppercased_anns]


def test_key_group():
    """Doc pipeline with keys group for annotation"""
    uppercaser = _Uppercaser(output_label="uppercased_sentence")
    step = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    pipeline = Pipeline(
        [step],
        input_keys=step.input_keys,
        output_keys=step.output_keys,
    )

    doc_pipeline = DocPipeline(pipeline=pipeline, labels_by_input_key={"SENTENCE": ["sentence"]})
    doc = _get_doc()
    doc_pipeline.run([doc])

    uppercased_anns = doc.anns.get(label="uppercased_sentence")
    for ann in uppercased_anns:
        assert ann.keys == {"UPPERCASE"}
