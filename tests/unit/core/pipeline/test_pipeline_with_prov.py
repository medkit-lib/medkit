from medkit.core import OperationDescription, generate_id
from medkit.core.pipeline import Pipeline, PipelineStep
from medkit.core.prov_tracer import ProvTracer


class _Segment:
    """Mock text segment"""

    def __init__(self, text):
        self.uid = generate_id()
        self.text = text
        self.attrs = []


class _Attribute:
    """Mock attribute"""

    def __init__(self, label, value):
        self.uid = generate_id()
        self.label = label
        self.value = value


_SENTENCES = [
    "This is a sentence",
    "This is another sentence",
    "This is the last sentence",
]


def _get_sentence_segments():
    return [_Segment(text=text) for text in _SENTENCES]


class _Uppercaser:
    """Mock processing operation uppercasing segments"""

    def __init__(self):
        self.uid = generate_id()
        self._prov_tracer = None

    @property
    def description(self):
        return OperationDescription(uid=self.uid, name="Uppercaser")

    def set_prov_tracer(self, prov_tracer):
        self._prov_tracer = prov_tracer

    def run(self, segments):
        uppercase_segments = []
        for segment in segments:
            uppercase_segment = _Segment(segment.text.upper())
            uppercase_segments.append(uppercase_segment)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(uppercase_segment, self.description, source_data_items=[segment])
        return uppercase_segments


class _Prefixer:
    """Mock processing operation prefixing segments"""

    def __init__(self, prefix):
        self.uid = generate_id()
        self.prefix = prefix

    @property
    def description(self):
        return OperationDescription(uid=self.uid, name="Prefixer")

    def set_prov_tracer(self, prov_tracer):
        self._prov_tracer = prov_tracer

    def run(self, segments):
        prefixed_segments = []
        for segment in segments:
            prefixed_segment = _Segment(self.prefix + segment.text)
            prefixed_segments.append(prefixed_segment)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(prefixed_segment, self.description, source_data_items=[segment])
        return prefixed_segments


class _AttributeAdder:
    """Mock processing operation adding attributes to existing segments"""

    def __init__(self, label):
        self.uid = generate_id()
        self.label = label

    @property
    def description(self):
        return OperationDescription(uid=self.uid, name="AttributeAdder")

    def set_prov_tracer(self, prov_tracer):
        self._prov_tracer = prov_tracer

    def run(self, segments):
        for segment in segments:
            attr = _Attribute(label=self.label, value=True)
            segment.attrs.append(attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(attr, self.description, source_data_items=[segment])


def test_single_step():
    """Minimalist pipeline with only one step"""
    uppercaser = _Uppercaser()
    step = PipelineStep(operation=uppercaser, input_keys=["SENTENCE"], output_keys=["UPPERCASE"])

    pipeline = Pipeline(steps=[step], input_keys=["SENTENCE"], output_keys=["UPPERCASE"])

    prov_tracer = ProvTracer()
    pipeline.set_prov_tracer(prov_tracer)
    sentence_segs = _get_sentence_segments()
    uppercased_segs = pipeline.run(sentence_segs)
    assert len(uppercased_segs) == len(sentence_segs)

    prov_tracer._graph.check_sanity()

    # check outer main provenance
    for uppercased_seg, sentence_seg in zip(uppercased_segs, sentence_segs):
        uppercased_seg_prov = prov_tracer.get_prov(uppercased_seg.uid)
        # operation is outer pipeline operation
        assert uppercased_seg_prov.op_desc == pipeline.description
        # uppercased segment has corresponding sentence segment as source
        assert uppercased_seg_prov.source_data_items == [sentence_seg]

        # sentence seg has stub provenance (no operation, no source)
        sentence_seg_prov = prov_tracer.get_prov(sentence_seg.uid)
        assert sentence_seg_prov.op_desc is None
        assert len(sentence_seg_prov.source_data_items) == 0

    # check inner sub provenance
    sub_tracer = prov_tracer.get_sub_prov_tracer(pipeline.uid)

    for uppercased_seg, sentence_seg in zip(uppercased_segs, sentence_segs):
        uppercased_seg_prov = sub_tracer.get_prov(uppercased_seg.uid)
        # operation is inner uppercaser operation
        assert uppercased_seg_prov.op_desc == uppercaser.description
        # uppercased segment has corresponding sentence segment as source
        assert uppercased_seg_prov.source_data_items == [sentence_seg]

        # sentence segment has stub provenance (no operation, no source)
        sentence_seg_prov = sub_tracer.get_prov(sentence_seg.uid)
        assert sentence_seg_prov.op_desc is None
        assert len(sentence_seg_prov.source_data_items) == 0


def test_multiple_steps():
    """Simple pipeline with 2 consecutive steps"""
    prefixer = _Prefixer(prefix="Hello! ")
    step_1 = PipelineStep(operation=prefixer, input_keys=["SENTENCE"], output_keys=["PREFIX"])

    uppercaser = _Uppercaser()
    step_2 = PipelineStep(operation=uppercaser, input_keys=["PREFIX"], output_keys=["UPPERCASE"])

    pipeline = Pipeline(steps=[step_1, step_2], input_keys=["SENTENCE"], output_keys=["UPPERCASE"])

    prov_tracer = ProvTracer()
    pipeline.set_prov_tracer(prov_tracer)
    sentence_segs = _get_sentence_segments()
    uppercased_segs = pipeline.run(sentence_segs)
    assert len(uppercased_segs) == len(sentence_segs)

    prov_tracer._graph.check_sanity()

    # check outer main provenance
    for uppercased_seg, sentence_seg in zip(uppercased_segs, sentence_segs):
        uppercased_seg_prov = prov_tracer.get_prov(uppercased_seg.uid)
        # operation is pipeline operation
        assert uppercased_seg_prov.op_desc == pipeline.description
        # uppercased segment has corresponding sentence as source
        assert uppercased_seg_prov.source_data_items == [sentence_seg]

    # check inner sub provenance
    sub_tracer = prov_tracer.get_sub_prov_tracer(pipeline.uid)

    for uppercased_seg, sentence_seg in zip(uppercased_segs, sentence_segs):
        uppercased_seg_prov = sub_tracer.get_prov(uppercased_seg.uid)
        # operation is inner uppercaser operation
        assert uppercased_seg_prov.op_desc == uppercaser.description
        # uppercased segment has corresponding prefixed segment as source
        assert len(uppercased_seg_prov.source_data_items) == 1
        prefixed_seg = uppercased_seg_prov.source_data_items[0]

        prefixed_seg_prov = sub_tracer.get_prov(prefixed_seg.uid)
        # operation is inner prefixer operation
        assert prefixed_seg_prov.op_desc == prefixer.description
        # prefixed segment has corresponding sentence as source
        assert prefixed_seg_prov.source_data_items == [sentence_seg]


def test_step_with_attributes():
    """Pipeline with a step adding attributes to existing segments instead of returning
    new segments
    """
    prefixer = _Prefixer(prefix="Hello! ")
    step_1 = PipelineStep(operation=prefixer, input_keys=["SENTENCE"], output_keys=["PREFIX"])

    uppercaser = _Uppercaser()
    step_2 = PipelineStep(operation=uppercaser, input_keys=["PREFIX"], output_keys=["UPPERCASE"])

    attribute_adder = _AttributeAdder(label="validated")
    step_3 = PipelineStep(operation=attribute_adder, input_keys=["UPPERCASE"], output_keys=[])

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prov_tracer = ProvTracer()
    pipeline.set_prov_tracer(prov_tracer)
    sentence_segs = _get_sentence_segments()
    uppercased_segs = pipeline.run(sentence_segs)
    assert len(uppercased_segs) == len(sentence_segs)

    prov_tracer._graph.check_sanity()

    # check outer main provenance
    for uppercased_seg, sentence_seg in zip(uppercased_segs, sentence_segs):
        assert len(uppercased_seg.attrs) == 1
        attr = uppercased_seg.attrs[0]
        attr_prov = prov_tracer.get_prov(attr.uid)
        # operation is outer pipeline operation
        assert attr_prov.op_desc == pipeline.description
        # attribute has corresponding sentence segment as source
        # (because it is what was given as input to the pipeline operation)
        assert attr_prov.source_data_items == [sentence_seg]

    # check inner sub provenance
    sub_tracer = prov_tracer.get_sub_prov_tracer(pipeline.uid)

    for uppercased_seg in uppercased_segs:
        assert len(uppercased_seg.attrs) == 1
        attr = uppercased_seg.attrs[0]
        attr_prov = sub_tracer.get_prov(attr.uid)
        # operation is inner attribute adder operation
        assert attr_prov.op_desc == attribute_adder.description
        # attribute has corresponding uppercased segment as source
        assert attr_prov.source_data_items == [uppercased_seg]


def test_nested_pipeline():
    """Pipeline with a step that is also a pipeline"""
    # build inner pipeline
    uppercaser = _Uppercaser()
    sub_step_1 = PipelineStep(
        operation=uppercaser,
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix_1 = "Hi! "
    prefixer_1 = _Prefixer(prefix_1)
    sub_step_2 = PipelineStep(
        operation=prefixer_1,
        input_keys=["UPPERCASE"],
        output_keys=["PREFIX"],
    )

    sub_pipeline = Pipeline(steps=[sub_step_1, sub_step_2], input_keys=["SENTENCE"], output_keys=["PREFIX"])

    # build main pipeline
    prefix_2 = "Hello! "
    prefixer_2 = _Prefixer(prefix_2)
    step_1 = PipelineStep(
        operation=prefixer_2,
        input_keys=["SENTENCE"],
        output_keys=["PREFIX"],
    )

    step_2 = PipelineStep(
        operation=sub_pipeline,
        input_keys=["PREFIX"],
        output_keys=["SUB_PIPELINE"],
    )

    pipeline = Pipeline(
        steps=[step_1, step_2],
        input_keys=["SENTENCE"],
        output_keys=["SUB_PIPELINE"],
    )

    prov_tracer = ProvTracer()
    pipeline.set_prov_tracer(prov_tracer)
    sentence_segs = _get_sentence_segments()
    output_segs = pipeline.run(sentence_segs)
    assert len(output_segs) == len(sentence_segs)

    prov_tracer._graph.check_sanity()

    # check outer main provenance
    for output_seg, sentence_seg in zip(output_segs, sentence_segs):
        output_seg_prov = prov_tracer.get_prov(output_seg.uid)
        # operation is main pipeline operation
        assert output_seg_prov.op_desc == pipeline.description
        # uppercased segment has corresponding sentence as source
        assert output_seg_prov.source_data_items == [sentence_seg]

    # check inner sub provenance
    sub_tracer = prov_tracer.get_sub_prov_tracer(pipeline.uid)
    for output_seg, sentence_seg in zip(output_segs, sentence_segs):
        # prov for result of sub pipeline
        prov = sub_tracer.get_prov(output_seg.uid)
        assert prov.op_desc == sub_pipeline.description

        # prov for result of prefixer before sub pipeline
        prov = sub_tracer.get_prov(prov.source_data_items[0].uid)
        assert prov.op_desc == prefixer_2.description
        assert len(prov.source_data_items) == 1

        # stub prov for input sentence segment
        prov = sub_tracer.get_prov(prov.source_data_items[0].uid)
        assert prov.data_item == sentence_seg
        assert prov.op_desc is None
        assert len(prov.source_data_items) == 0

    # check innermost sub provenance
    sub_tracer = sub_tracer.get_sub_prov_tracer(sub_pipeline.uid)
    for output_seg, _ in zip(output_segs, sentence_segs):
        # prov for result of prefixer inside sub pipeline
        prov = sub_tracer.get_prov(output_seg.uid)
        assert prov.op_desc == prefixer_1.description
        assert len(prov.source_data_items) == 1

        # prov for result of uppercaser inside sub pipeline
        prov = sub_tracer.get_prov(prov.source_data_items[0].uid)
        assert prov.op_desc == uppercaser.description
        assert len(prov.source_data_items) == 1

        # stub prov for result of prefixer before sub pipeline
        prov = sub_tracer.get_prov(prov.source_data_items[0].uid)
        assert prov.op_desc is None
        assert len(prov.source_data_items) == 0
