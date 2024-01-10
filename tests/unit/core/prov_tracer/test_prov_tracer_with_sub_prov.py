from medkit.core import OperationDescription, generate_id
from medkit.core.prov_tracer import ProvTracer
from tests.unit.core.prov_tracer._common import (
    Merger,
    Prefixer,
    Splitter,
    get_text_items,
)


class _PrefixerWrapper:
    def __init__(self, prov_tracer):
        self.uid = generate_id()
        self.prefixer = Prefixer()
        self.prov_tracer = prov_tracer
        self.sub_prov_tracer = ProvTracer(prov_tracer.store)
        self.prefixer.prov_tracer = self.sub_prov_tracer
        self.description = OperationDescription(uid=self.uid, name="PrefixerWrapper")

    def run(self, input_items):
        output_items = self.prefixer.prefix(input_items)

        self.prov_tracer.add_prov_from_sub_tracer(output_items, self.description, self.sub_prov_tracer)

        return output_items


def test_single_operation():
    """Composite operation using only one operation"""
    tracer = ProvTracer()
    wrapper = _PrefixerWrapper(tracer)
    input_items = get_text_items(2)
    output_items = wrapper.run(input_items)

    tracer._graph.check_sanity()

    # check outer main provenance
    # must have prov for each input item and each output item
    assert len(tracer.get_provs()) == len(input_items) + len(output_items)

    for input_item, output_item in zip(input_items, output_items):
        input_prov = tracer.get_prov(input_item.uid)
        assert input_prov.data_item == input_item
        assert input_prov.op_desc is None
        assert len(input_prov.source_data_items) == 0
        # input item has corresponding output item as derived
        assert input_prov.derived_data_items == [output_item]

        output_prov = tracer.get_prov(output_item.uid)
        assert output_prov.data_item == output_item
        # operation is outer wrapper operation
        assert output_prov.op_desc == wrapper.description
        # output item has corresponding input item as source
        assert output_prov.source_data_items == [input_item]
        assert len(output_prov.derived_data_items) == 0

    # check inner sub provenance
    assert tracer.has_sub_prov_tracer(wrapper.uid)
    assert len(tracer.get_sub_prov_tracers()) == 1
    sub_tracer = tracer.get_sub_prov_tracer(wrapper.uid)
    # must have prov for each input item and each output item
    assert len(sub_tracer.get_provs()) == len(input_items) + len(output_items)

    for input_item, output_item in zip(input_items, output_items):
        input_prov = sub_tracer.get_prov(input_item.uid)
        assert input_prov.data_item == input_item
        assert input_prov.op_desc is None
        assert len(input_prov.source_data_items) == 0
        # input item has corresponding output item as derived
        assert input_prov.derived_data_items == [output_item]

        output_prov = sub_tracer.get_prov(output_item.uid)
        assert output_prov.data_item == output_item
        # operation is inner prefixer
        assert output_prov.op_desc == wrapper.prefixer.description
        # output item has corresponding input item as source
        assert output_prov.source_data_items == [input_item]
        assert len(output_prov.derived_data_items) == 0


class _DoublePrefixerWrapper:
    def __init__(self, prov_tracer):
        self.uid = generate_id()
        self.prefixer_1 = Prefixer()
        self.prefixer_2 = Prefixer()
        self.prov_tracer = prov_tracer
        self.sub_prov_tracer = ProvTracer(prov_tracer.store)
        self.prefixer_1.prov_tracer = self.sub_prov_tracer
        self.prefixer_2.prov_tracer = self.sub_prov_tracer
        self.description = OperationDescription(uid=self.uid, name="DoublePrefixerWrapper")

    def run(self, input_items):
        intermediate_items = self.prefixer_1.prefix(input_items)
        output_items = self.prefixer_2.prefix(intermediate_items)

        self.prov_tracer.add_prov_from_sub_tracer(output_items, self.description, self.sub_prov_tracer)

        return output_items


def test_intermediate_operation():
    """Composite operation using 2 consecutive operations, only the output of the
    2d operation is returned
    """
    tracer = ProvTracer()
    wrapper = _DoublePrefixerWrapper(tracer)
    input_items = get_text_items(2)
    output_items = wrapper.run(input_items)

    tracer._graph.check_sanity()

    # check outer main provenance
    # must have prov for each input item and each output item
    assert len(tracer.get_provs()) == len(input_items) + len(output_items)

    for input_item, output_item in zip(input_items, output_items):
        input_prov = tracer.get_prov(input_item.uid)
        # input item has corresponding output item as derived
        assert input_prov.derived_data_items == [output_item]

        output_prov = tracer.get_prov(output_item.uid)
        # operation is outer wrapper operation
        assert output_prov.op_desc == wrapper.description
        # output item has corresponding input item as source
        assert output_prov.source_data_items == [input_item]

    # check inner sub provenance
    sub_tracer = tracer.get_sub_prov_tracer(wrapper.uid)
    # must have prov for each input item, each intermediate item and each output item
    assert len(sub_tracer.get_provs()) == len(input_items) + 2 * len(output_items)

    for input_item, output_item in zip(input_items, output_items):
        assert sub_tracer.has_prov(input_item.uid)

        output_prov = sub_tracer.get_prov(output_item.uid)
        # operation is inner 1st prefixer
        assert output_prov.op_desc == wrapper.prefixer_2.description
        # output item has 1 intermediate item as source
        assert len(output_prov.source_data_items) == 1
        intermediate_item = output_prov.source_data_items[0]

        intermediate_prov = sub_tracer.get_prov(intermediate_item.uid)
        # operation is inner 2d prefixer
        assert intermediate_prov.op_desc == wrapper.prefixer_1.description
        # intermediate item has corresponding input item as source
        assert intermediate_prov.source_data_items == [input_item]
        # intermediate item has corresponding output item as derived
        assert intermediate_prov.derived_data_items == [output_item]

        input_prov = sub_tracer.get_prov(input_item.uid)
        # input item has corresponding intermediate item as derived
        assert input_prov.derived_data_items == [intermediate_item]


class _PrefixerMergerWrapper:
    def __init__(self, prov_tracer):
        self.uid = generate_id()
        self.prefixer = Prefixer()
        self.merger = Merger()
        self.prov_tracer = prov_tracer
        self.sub_prov_tracer = ProvTracer(prov_tracer.store)
        self.prefixer.prov_tracer = self.sub_prov_tracer
        self.merger.prov_tracer = self.sub_prov_tracer
        self.description = OperationDescription(uid=self.uid, name="PrefixerMergerWrapper")

    def run(self, input_items):
        intermediate_items = self.prefixer.prefix(input_items)
        output_item = self.merger.merge(intermediate_items)

        self.prov_tracer.add_prov_from_sub_tracer([output_item], self.description, self.sub_prov_tracer)

        return output_item


def test_multi_input_operation():
    """Composite operation using an operation deriving one item from multiple input
    (merger)
    """
    tracer = ProvTracer()
    wrapper = _PrefixerMergerWrapper(tracer)
    input_items = get_text_items(2)
    output_item = wrapper.run(input_items)

    tracer._graph.check_sanity()

    # check outer main provenance
    # must have prov for each input item and each output item
    assert len(tracer.get_provs()) == len(input_items) + 1

    output_prov = tracer.get_prov(output_item.uid)
    # operation is outer wrapper operation
    assert output_prov.op_desc == wrapper.description
    # output item has all input items as sources
    assert output_prov.source_data_items == input_items

    # check inner sub provenance
    sub_tracer = tracer.get_sub_prov_tracer(wrapper.uid)
    # must have prov for each input item, each prefixed item and each output item
    nb_prefixed_items = len(input_items)
    assert len(sub_tracer.get_provs()) == len(input_items) + nb_prefixed_items + 1
    # merged item has all prefixed items as sources
    merged_prov = sub_tracer.get_prov(output_item.uid)
    assert merged_prov.op_desc == wrapper.merger.description
    assert len(merged_prov.source_data_items) == nb_prefixed_items

    for prefixed_item, input_item in zip(merged_prov.source_data_items, input_items):
        prefixed_prov = sub_tracer.get_prov(prefixed_item.uid)
        # operation is inner prefixer
        assert prefixed_prov.op_desc == wrapper.prefixer.description
        # prefixed item has corresponding input item as source
        assert prefixed_prov.source_data_items == [input_item]


class _SplitterPrefixerWrapper:
    def __init__(self, prov_tracer):
        self.uid = generate_id()
        self.splitter = Splitter()
        self.prefixer = Prefixer()
        self.prov_tracer = prov_tracer
        self.sub_prov_tracer = ProvTracer(prov_tracer.store)
        self.splitter.prov_tracer = self.sub_prov_tracer
        self.prefixer.prov_tracer = self.sub_prov_tracer
        self.description = OperationDescription(uid=self.uid, name="SplitterPrefixerMWrapper")

    def run(self, input_items):
        intermediate_items = self.splitter.split(input_items)
        output_items = self.prefixer.prefix(intermediate_items)

        self.prov_tracer.add_prov_from_sub_tracer(output_items, self.description, self.sub_prov_tracer)

        return output_items


def test_multi_output_operation():
    """Composite operation using an operation derived several items from single
    input item (splitter)
    """
    tracer = ProvTracer()
    wrapper = _SplitterPrefixerWrapper(tracer)
    input_items = get_text_items(2)
    output_items = wrapper.run(input_items)

    tracer._graph.check_sanity()

    # check outer main provenance
    # must have prov for each input item and each output item
    assert len(tracer.get_provs()) == len(input_items) + len(output_items)

    for i, output_item in enumerate(output_items):
        input_item = input_items[i // 2]
        # output item has corresponding input item as source
        output_prov = tracer.get_prov(output_item.uid)
        # operation is outer wrapper operation
        assert output_prov.op_desc == wrapper.description
        assert output_prov.source_data_items == [input_item]

    # check inner sub provenance
    sub_tracer = tracer.get_sub_prov_tracer(wrapper.uid)
    # must have prov for each input item, each prefixed item and each output item
    nb_split_items = 2 * len(input_items)
    assert len(sub_tracer.get_provs()) == len(input_items) + nb_split_items + len(output_items)

    for i, prefixed_item in enumerate(output_items):
        input_item = input_items[i // 2]

        prefixed_prov = sub_tracer.get_prov(prefixed_item.uid)
        assert prefixed_prov.op_desc == wrapper.prefixer.description
        # prefixed item has one split item as source
        assert len(prefixed_prov.source_data_items) == 1

        split_item = prefixed_prov.source_data_items[0]
        split_prov = sub_tracer.get_prov(split_item.uid)
        # operation is inner splitter
        assert split_prov.op_desc == wrapper.splitter.description
        # split item has its corresponding input item as source
        assert split_prov.source_data_items == [input_item]


class _BranchedPrefixerWrapper:
    def __init__(self, prov_tracer):
        self.uid = generate_id()
        self.prefixer_1 = Prefixer()
        self.prefixer_2 = Prefixer()
        self.prov_tracer = prov_tracer
        self.sub_prov_tracer = ProvTracer(prov_tracer.store)
        self.prefixer_1.prov_tracer = self.sub_prov_tracer
        self.prefixer_2.prov_tracer = self.sub_prov_tracer
        self.description = OperationDescription(uid=self.uid, name="BrancherPrefixerWrapper")

    def run(self, input_items):
        prefixed_items = self.prefixer_1.prefix(input_items)
        double_prefixed_items = self.prefixer_2.prefix(prefixed_items)

        output_items = prefixed_items + double_prefixed_items
        self.prov_tracer.add_prov_from_sub_tracer(output_items, self.description, self.sub_prov_tracer)

        return prefixed_items, double_prefixed_items


def test_operation_reusing_output():
    """Composite operation using 2 operations, with 2 outputs, the 2d output being
    the result of the 2d operation applied on the 1st output (make sure we don't
    try to add the same item twice in the main graph)
    """
    tracer = ProvTracer()
    wrapper = _BranchedPrefixerWrapper(tracer)
    input_items = get_text_items(2)
    prefixed_items, double_prefixed_items = wrapper.run(input_items)

    tracer._graph.check_sanity()

    # check outer main provenance
    # must have prov for each input item and each output prefixed_items
    assert len(tracer.get_provs()) == len(input_items) + len(prefixed_items) + len(double_prefixed_items)

    for input_item, prefixed_item in zip(input_items, prefixed_items):
        prefixed_prov = tracer.get_prov(prefixed_item.uid)
        # operation is outer wrapper operation
        assert prefixed_prov.op_desc == wrapper.description
        # prefixed item has corresponding input item as source
        assert prefixed_prov.source_data_items == [input_item]
    for input_item, double_prefixed_item in zip(input_items, prefixed_items):
        double_prefixed_prov = tracer.get_prov(double_prefixed_item.uid)
        # operation is outer wrapper operation
        assert double_prefixed_prov.op_desc == wrapper.description
        # prefixed item has corresponding input item as source
        assert double_prefixed_prov.source_data_items == [input_item]

    # check inner sub provenance
    sub_tracer = tracer.get_sub_prov_tracer(wrapper.uid)
    # must have prov for each input item, each prefixed item and each double prefixed item
    assert len(sub_tracer.get_provs()) == len(input_items) + len(prefixed_items) + len(double_prefixed_items)

    for input_item, prefixed_item in zip(input_items, prefixed_items):
        prefixed_prov = sub_tracer.get_prov(prefixed_item.uid)
        # operation is inner 1st prefixer
        assert prefixed_prov.op_desc == wrapper.prefixer_1.description
        # prefixed item has corresponding input item as source
        assert prefixed_prov.source_data_items == [input_item]

    for prefixed_item, double_prefixed_item in zip(prefixed_items, double_prefixed_items):
        double_prefixed_prov = sub_tracer.get_prov(double_prefixed_item.uid)
        # operation is inner 2d prefixer
        assert double_prefixed_prov.op_desc == wrapper.prefixer_2.description
        # double prefixed item has corresponding prefixed item as source
        assert double_prefixed_prov.source_data_items == [prefixed_item]


def test_consecutive_calls():
    """Make sure add_prov_from_sub_tracer can be called several times"""
    tracer = ProvTracer()
    wrapper = _DoublePrefixerWrapper(tracer)
    input_items_1 = get_text_items(2)
    output_items_1 = wrapper.run(input_items_1)
    input_items_2 = get_text_items(2)
    output_items_2 = wrapper.run(input_items_2)
    input_items = input_items_1 + input_items_2
    output_items = output_items_1 + output_items_2

    tracer._graph.check_sanity()

    # check outer main provenance
    # must have prov for each input item and each output item
    assert len(tracer.get_provs()) == len(input_items) + len(output_items)

    # check inner sub provenance
    sub_tracer = tracer.get_sub_prov_tracer(wrapper.uid)
    # must have prov for each input item, each intermediate item and each output item
    assert len(sub_tracer.get_provs()) == len(input_items) + 2 * len(output_items)


class _NestedWrapper:
    def __init__(self, prov_tracer):
        self.uid = generate_id()
        self.prov_tracer = prov_tracer
        self.sub_prov_tracer = ProvTracer(prov_tracer.store)
        self.sub_wrapper_1 = _DoublePrefixerWrapper(self.sub_prov_tracer)
        self.sub_wrapper_2 = _DoublePrefixerWrapper(self.sub_prov_tracer)
        self.description = OperationDescription(uid=self.uid, name="NestedWrapper")

    def run(self, input_items):
        output_items = self.sub_wrapper_1.run(input_items)
        output_items += self.sub_wrapper_2.run(input_items)

        self.prov_tracer.add_prov_from_sub_tracer(output_items, self.description, self.sub_prov_tracer)

        return output_items


def test_nested():
    """Composite operation using 2 composite operations"""
    tracer = ProvTracer()
    wrapper = _NestedWrapper(tracer)
    input_items = get_text_items(2)
    prefixed_items = wrapper.run(input_items)

    tracer._graph.check_sanity()

    # check outer main provenance
    input_item = input_items[0]
    prefixed_item = prefixed_items[0]
    prov = tracer.get_prov(prefixed_item.uid)
    assert prov.op_desc == wrapper.description
    assert prov.source_data_items == [input_item]

    # check inner sub provenance
    assert len(tracer.get_sub_prov_tracers()) == 1
    sub_tracer = tracer.get_sub_prov_tracer(wrapper.uid)
    prov = sub_tracer.get_prov(prefixed_item.uid)
    assert prov.op_desc == wrapper.sub_wrapper_1.description
    assert prov.source_data_items == [input_item]

    # check innermost "sub sub" provenance
    assert len(sub_tracer.get_sub_prov_tracers()) == 2
    sub_sub_tracer_1 = sub_tracer.get_sub_prov_tracer(wrapper.sub_wrapper_1.uid)
    prov = sub_sub_tracer_1.get_prov(prefixed_item.uid)
    assert prov.op_desc == wrapper.sub_wrapper_1.prefixer_2.description
    intermediate_item = prov.source_data_items[0]
    prov = sub_sub_tracer_1.get_prov(intermediate_item.uid)
    assert prov.op_desc == wrapper.sub_wrapper_1.prefixer_1.description
    assert prov.source_data_items == [input_item]
