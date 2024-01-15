import re

import pytest

from medkit.core.pipeline import Pipeline, PipelineStep


class _Segment:
    """Mock text segment"""

    def __init__(self, text):
        self.text = text
        self.attrs = []


class _Attribute:
    """Mock attribute"""

    def __init__(self, label, value):
        self.label = label
        self.value = value


_SENTENCES = [
    "This is a sentence",
    "This is another sentence",
    "This is the last sentence",
]


def _get_sentence_segments():
    return [_Segment(text) for text in _SENTENCES]


class _Uppercaser:
    """Mock processing operation uppercasing segments"""

    def run(self, segments):
        uppercase_segments = []
        for segment in segments:
            uppercase_segment = _Segment(segment.text.upper())
            uppercase_segments.append(uppercase_segment)
        return uppercase_segments


class _Prefixer:
    """Mock processing operation prefixing segments"""

    def __init__(self, prefix):
        self.prefix = prefix

    def run(self, segments):
        prefixed_segments = []
        for segment in segments:
            prefixed_segment = _Segment(self.prefix + segment.text)
            prefixed_segments.append(prefixed_segment)
        return prefixed_segments


class _Splitter:
    """Mock processing operation splitting segments"""

    def run(self, segments):
        left_segments = []
        right_segments = []
        for segment in segments:
            half = len(segment.text) // 2
            left_segment = _Segment(segment.text[:half])
            left_segments.append(left_segment)
            right_segment = _Segment(segment.text[half:])
            right_segments.append(right_segment)
        return left_segments, right_segments


class _Merger:
    """Mock processing operation merging segments"""

    def run(self, left_segments, right_segments):
        merged_segments = []
        for left_segment, right_segment in zip(left_segments, right_segments):
            merged_segment = _Segment(left_segment.text + right_segment.text)
            merged_segments.append(merged_segment)
        return merged_segments


class _KeywordMatcher:
    """Mock processing operation finding exact keyword matches"""

    def __init__(self, keywords):
        self.keywords = keywords

    def run(self, segments):
        entities = []
        for segment in segments:
            for keyword in self.keywords:
                match = re.search(keyword, segment.text)
                if match is None:
                    continue
                entity = _Segment(match.group())
                entities.append(entity)
        return entities


class _AttributeAdder:
    """Mock processing operation adding attributes to existing segments"""

    def __init__(self, label):
        self.label = label

    def run(self, segments):
        for segment in segments:
            segment.attrs.append(_Attribute(label=self.label, value=True))


def test_single_step():
    """Minimalist pipeline with only one step"""
    step = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    pipeline = Pipeline(steps=[step], input_keys=["SENTENCE"], output_keys=["UPPERCASE"])

    sentence_segs = _get_sentence_segments()
    uppercased_segs = pipeline.run(sentence_segs)

    # operation was properly called to generate new data item
    assert [a.text.upper() for a in sentence_segs] == [a.text for a in uppercased_segs]


def test_multiple_steps():
    """Simple pipeline with 2 consecutive steps"""
    step_1 = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix = "Hello! "
    step_2 = PipelineStep(
        operation=_Prefixer(prefix=prefix),
        input_keys=["UPPERCASE"],
        output_keys=["PREFIX"],
    )

    pipeline = Pipeline(steps=[step_1, step_2], input_keys=["SENTENCE"], output_keys=["PREFIX"])

    sentence_segs = _get_sentence_segments()
    prefixed_uppercased_segs = pipeline.run(sentence_segs)

    # operations were properly called and in the correct order to generate new data items
    expected_texts = [prefix + a.text.upper() for a in sentence_segs]
    assert [a.text for a in prefixed_uppercased_segs] == expected_texts


def test_multiple_steps_with_same_output_key():
    """Pipeline with 2 step using the same output key, and another step
    using it as input
    """
    prefix_1 = "Hello! "
    _prefixer_1 = _Prefixer(prefix_1)
    step_1 = PipelineStep(
        operation=_prefixer_1,
        input_keys=["SENTENCE"],
        output_keys=["PREFIX"],
    )

    prefix_2 = "Hi! "
    _prefixer_2 = _Prefixer(prefix_2)
    step_2 = PipelineStep(
        operation=_prefixer_2,
        input_keys=["SENTENCE"],
        output_keys=["PREFIX"],
    )

    step_3 = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["PREFIX"],
        output_keys=["UPPERCASE"],
    )

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    sentence_segs = _get_sentence_segments()
    uppercased_segs = pipeline.run(sentence_segs)

    # operations were properly called in the correct order
    expected_texts = [(prefix_1 + a.text).upper() for a in sentence_segs] + [
        (prefix_2 + a.text).upper() for a in sentence_segs
    ]
    assert [a.text for a in uppercased_segs] == expected_texts


def test_multiple_steps_with_same_input_key():
    """Pipeline with 2 step using the same input key,
    which is also the output key of a previous step
    """
    step_1 = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix_1 = "Hello! "
    _prefixer_2 = _Prefixer(prefix_1)
    step_2 = PipelineStep(
        operation=_prefixer_2,
        input_keys=["UPPERCASE"],
        output_keys=["PREFIX_1"],
    )

    prefix_2 = "Hello! "
    _prefixer_2 = _Prefixer(prefix_2)
    step_3 = PipelineStep(
        operation=_prefixer_2,
        input_keys=["UPPERCASE"],
        output_keys=["PREFIX_2"],
    )

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        input_keys=["SENTENCE"],
        output_keys=["PREFIX_1", "PREFIX_2"],
    )

    sentence_segs = _get_sentence_segments()
    prefixed_uppercased_segs_1, prefixed_uppercased_segs_2 = pipeline.run(sentence_segs)

    # operations were properly called in the correct order
    expected_texts_1 = [prefix_1 + a.text.upper() for a in sentence_segs]
    assert [a.text for a in prefixed_uppercased_segs_1] == expected_texts_1
    expected_texts_2 = [prefix_2 + a.text.upper() for a in sentence_segs]
    assert [a.text for a in prefixed_uppercased_segs_2] == expected_texts_2


def test_step_with_multiple_outputs():
    """Pipeline with a step having more than 1 output"""
    step_1 = PipelineStep(
        operation=_Splitter(),
        input_keys=["SENTENCE"],
        output_keys=["SPLIT_LEFT", "SPLIT_RIGHT"],
    )

    step_2 = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["SPLIT_LEFT"],
        output_keys=["UPPERCASE"],
    )

    prefix = "Hello! "
    step_3 = PipelineStep(
        operation=_Prefixer(prefix=prefix),
        input_keys=["SPLIT_RIGHT"],
        output_keys=["PREFIX"],
    )

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE", "PREFIX"],
    )

    sentence_segs = _get_sentence_segments()
    uppercased_left_segs, prefixed_right_segs = pipeline.run(sentence_segs)

    # operations were properly called in the correct order
    expected_texts = [a.text[: len(a.text) // 2].upper() for a in sentence_segs]
    assert [a.text for a in uppercased_left_segs] == expected_texts
    expected_texts = [prefix + a.text[len(a.text) // 2 :] for a in sentence_segs]
    assert [a.text for a in prefixed_right_segs] == expected_texts


def test_step_with_multiple_inputs():
    """Pipeline with a step having more than 1 input"""
    step_1 = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["SENTENCE"],
        output_keys=["UPPERCASE"],
    )

    prefix = "Hello! "
    step_2 = PipelineStep(
        operation=_Prefixer(prefix=prefix),
        input_keys=["SENTENCE"],
        output_keys=["PREFIX"],
    )

    step_3 = PipelineStep(
        operation=_Merger(),
        input_keys=["UPPERCASE", "PREFIX"],
        output_keys=["MERGE"],
    )

    pipeline = Pipeline(steps=[step_1, step_2, step_3], input_keys=["SENTENCE"], output_keys=["MERGE"])

    sentence_segs = _get_sentence_segments()
    merged_segs = pipeline.run(sentence_segs)

    # operations were properly called in the correct order
    expected_texts = [a.text.upper() + prefix + a.text for a in sentence_segs]
    assert [a.text for a in merged_segs] == expected_texts


def test_step_with_no_output():
    """Pipeline with a step having no output, because it modifies the data items
    it receives by adding attributes to them
    """
    step_1 = PipelineStep(
        operation=_AttributeAdder(label="validated"),
        input_keys=["SENTENCE"],
        output_keys=[],
    )

    pipeline = Pipeline(steps=[step_1], input_keys=["SENTENCE"], output_keys=[])

    sentence_segs = _get_sentence_segments()
    return_value = pipeline.run(sentence_segs)
    assert return_value is None

    # make sure attributes were added
    for segment in sentence_segs:
        assert len(segment.attrs) == 1
        attr = segment.attrs[0]
        assert attr.label == "validated"
        assert attr.value is True


def test_step_with_different_output_length():
    """Simple pipeline with 2 consecutive steps. 1st step returns a number of data items
    different from the number of data items it received as input
    """
    step_1 = PipelineStep(
        operation=_KeywordMatcher(keywords=["sentence", "another"]),
        input_keys=["SENTENCE"],
        output_keys=["KEYWORD_MATCH"],
    )

    step_2 = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["KEYWORD_MATCH"],
        output_keys=["UPPERCASE"],
    )

    pipeline = Pipeline(steps=[step_1, step_2], input_keys=["SENTENCE"], output_keys=["UPPERCASE"])

    sentence_segs = _get_sentence_segments()
    entities = pipeline.run(sentence_segs)
    assert len(entities) == 4


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

    sentence_segs = _get_sentence_segments()
    output_segs = pipeline.run(sentence_segs)

    # operations were properly called and in the correct order to generate new annotations
    expected_texts = [prefix_1 + (prefix_2 + a.text).upper() for a in sentence_segs]
    assert [a.text for a in output_segs] == expected_texts


def test_sanity_check():
    """Sanity checks"""
    uppercaser = _Uppercaser()
    prefixer = _Prefixer(prefix="prefix ")

    steps_1 = [
        PipelineStep(uppercaser, input_keys=["SENTENCE"], output_keys=["UPPERCASE"]),
        PipelineStep(prefixer, input_keys=["UPPERCASE"], output_keys=["PREFIX"]),
    ]

    # valid pipeline should not raise
    pipeline_1 = Pipeline(steps=steps_1, input_keys=["SENTENCE"], output_keys=["PREFIX"])
    pipeline_1.check_sanity()

    # pipeline input key not corresponding to any step input key
    pipeline_2 = Pipeline(steps=steps_1, input_keys=["WRONG_KEY"], output_keys=["PREFIX"])
    with pytest.raises(
        Exception,
        match="Pipeline input key WRONG_KEY does not correspond to any step input key",
    ):
        pipeline_2.check_sanity()

    # pipeline output key not corresponding to any step input key
    pipeline_3 = Pipeline(steps=steps_1, input_keys=["SENTENCE"], output_keys=["WRONG_KEY"])
    with pytest.raises(
        Exception,
        match=("Pipeline output key WRONG_KEY does not correspond to any step output key"),
    ):
        pipeline_3.check_sanity()

    # step input key not corresponding to any step or pipeline input key
    steps_2 = [
        PipelineStep(uppercaser, input_keys=["SENTENCE"], output_keys=["UPPERCASE"]),
        PipelineStep(prefixer, input_keys=["WRONG_KEY"], output_keys=["PREFIX"]),
    ]
    pipeline_4 = Pipeline(steps=steps_2, input_keys=["SENTENCE"], output_keys=["PREFIX"])
    with pytest.raises(
        Exception,
        match=("Step input key WRONG_KEY does not correspond to any step output key nor any pipeline input key"),
    ):
        pipeline_4.check_sanity()

    # step input key not available yet
    steps_3 = list(reversed(steps_1))
    pipeline_5 = Pipeline(steps=steps_3, input_keys=["SENTENCE"], output_keys=["PREFIX"])
    with pytest.raises(Exception, match="Step input key UPPERCASE is not available yet at this step"):
        pipeline_5.check_sanity()


def test_input_aggregation():
    """Step aggregating data for multiple keys and passing it to one input"""
    step = PipelineStep(
        operation=_Uppercaser(),
        input_keys=["SENTENCE_1", "SENTENCE_2"],
        output_keys=["UPPERCASE"],
        aggregate_input_keys=True,
    )

    pipeline = Pipeline(steps=[step], input_keys=["SENTENCE_1", "SENTENCE_2"], output_keys=["UPPERCASE"])

    sentence_segs_1 = _get_sentence_segments()
    sentence_segs_2 = _get_sentence_segments()
    uppercased_segs = pipeline.run(sentence_segs_1, sentence_segs_2)

    # operation was properly called to generate new data item
    assert [a.text.upper() for a in sentence_segs_1 + sentence_segs_2] == [a.text for a in uppercased_segs]
