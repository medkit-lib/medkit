import pytest

from medkit.core import generate_id
from medkit.core._prov_graph import ProvGraph, ProvNode


def test_basic():
    """Basic usage"""
    graph = ProvGraph()

    # add 1st node
    data_item_id_1 = generate_id()
    op_id_1 = generate_id()
    graph.add_node(data_item_id_1, op_id_1, source_ids=[])

    # add 2d node with same operation as 1st node
    data_item_id_2 = generate_id()
    graph.add_node(data_item_id_2, op_id_1, source_ids=[])

    # add 3rd node with 1st node as source
    data_item_id_3 = generate_id()
    op_id_2 = generate_id()
    graph.add_node(data_item_id_3, op_id_2, source_ids=[data_item_id_1])

    # retrieve all nodes and check them
    nodes = graph.get_nodes()
    assert len(nodes) == 3
    node_1, node_2, node_3 = nodes

    assert node_1.data_item_id == data_item_id_1
    assert node_1.operation_id == op_id_1
    assert len(node_1.source_ids) == 0
    # 3rd node was automatically added to derived items of 1st node
    assert node_1.derived_ids == [data_item_id_3]

    assert node_2.data_item_id == data_item_id_2
    assert node_2.operation_id == op_id_1
    assert len(node_2.source_ids) == 0
    assert len(node_2.derived_ids) == 0

    assert node_3.data_item_id == data_item_id_3
    assert node_3.operation_id == op_id_2
    assert node_3.source_ids == [data_item_id_1]
    assert len(node_3.derived_ids) == 0

    # test other node accessors
    assert graph.has_node(data_item_id_1)
    assert graph.get_node(data_item_id_1) == node_1
    non_existent_id = generate_id()
    assert not graph.has_node(non_existent_id)


def test_multiple_derived():
    """More complicated derivations"""
    graph = ProvGraph()

    # add 1st and 2d node
    data_item_id_1 = generate_id()
    graph.add_node(
        data_item_id_1,
        operation_id=generate_id(),
        source_ids=[],
    )
    data_item_id_2 = generate_id()
    graph.add_node(
        data_item_id_2,
        operation_id=generate_id(),
        source_ids=[],
    )

    # add 3d node derived from 1st and 2d
    data_item_id_3 = generate_id()
    graph.add_node(
        data_item_id_3,
        operation_id=generate_id(),
        source_ids=[data_item_id_1, data_item_id_2],
    )

    # add 4th node derived from 1st
    data_item_id_4 = generate_id()
    graph.add_node(
        data_item_id_4,
        operation_id=generate_id(),
        source_ids=[data_item_id_1],
    )

    # check source/derived for all nodes
    node_1 = graph.get_node(data_item_id_1)
    assert len(node_1.source_ids) == 0
    assert node_1.derived_ids == [data_item_id_3, data_item_id_4]

    node_2 = graph.get_node(data_item_id_2)
    assert len(node_2.source_ids) == 0
    assert node_2.derived_ids == [data_item_id_3]

    node_3 = graph.get_node(data_item_id_3)
    assert node_3.source_ids == [data_item_id_1, data_item_id_2]
    assert len(node_3.derived_ids) == 0

    node_4 = graph.get_node(data_item_id_4)
    assert node_4.source_ids == [data_item_id_1]
    assert len(node_4.derived_ids) == 0


def test_stub_node():
    """Handling of stub nodes for unknown data items"""
    graph = ProvGraph()

    # add node created from unknown data item
    data_item_1 = generate_id()
    data_item_2 = generate_id()
    graph.add_node(
        data_item_1,
        operation_id=generate_id(),
        source_ids=[data_item_2],
    )

    node_1 = graph.get_node(data_item_1)
    assert node_1.source_ids == [data_item_2]

    # stub node automatically created
    node_2 = graph.get_node(data_item_2)
    assert node_2.operation_id is None
    assert node_2.source_ids == []
    assert node_2.derived_ids == [data_item_1]

    # create node for previously unknown data item
    op_id = generate_id()
    graph.add_node(
        data_item_2,
        operation_id=op_id,
        source_ids=[],
    )
    node_2 = graph.get_node(data_item_2)
    assert node_2.operation_id == op_id
    assert node_2.source_ids == []
    assert node_2.derived_ids == [data_item_1]


def test_init_from_nodes():
    """Initialize a graph with a list of nodes"""
    data_item_id_1 = generate_id()
    data_item_id_2 = generate_id()

    node_1 = ProvNode(
        data_item_id=data_item_id_1,
        operation_id=generate_id(),
        source_ids=[],
        derived_ids=[data_item_id_2],
    )

    node_2 = ProvNode(
        data_item_id=data_item_id_2,
        operation_id=generate_id(),
        source_ids=[data_item_id_1],
        derived_ids=[],
    )

    graph = ProvGraph(nodes=[node_1, node_2])
    assert graph.get_nodes() == [node_1, node_2]


def _gen_simple_graph():
    graph = ProvGraph()

    data_item_id_1 = generate_id()
    graph.add_node(
        data_item_id=data_item_id_1,
        operation_id=generate_id(),
        source_ids=[],
    )

    data_item_id_2 = generate_id()
    graph.add_node(
        data_item_id=data_item_id_2,
        operation_id=generate_id(),
        source_ids=[data_item_id_1],
    )

    data_item_id_3 = generate_id()
    graph.add_node(
        data_item_id=data_item_id_3,
        operation_id=generate_id(),
        source_ids=[data_item_id_2],
    )

    return graph


def test_sub_graph_basic():
    """Basic behavior of graph with 2 sub graphs"""
    graph = _gen_simple_graph()

    # add a sub graph corresponding to an operation in the main
    node_1 = graph.get_nodes()[0]
    sub_graph_1 = _gen_simple_graph()
    graph.add_sub_graph(node_1.operation_id, sub_graph_1)
    assert graph.get_sub_graphs() == [sub_graph_1]
    assert graph.has_sub_graph(node_1.operation_id)
    assert graph.get_sub_graph(node_1.operation_id) == sub_graph_1

    # add another sub graph for a different operation
    node_2 = graph.get_nodes()[1]
    sub_graph_2 = _gen_simple_graph()
    graph.add_sub_graph(node_2.operation_id, sub_graph_2)
    assert graph.get_sub_graphs() == [sub_graph_1, sub_graph_2]
    assert graph.has_sub_graph(node_2.operation_id)
    assert graph.get_sub_graph(node_2.operation_id) == sub_graph_2

    # add a sub graph for an operation not in the main graph
    sub_graph_3 = _gen_simple_graph()
    graph.add_sub_graph(generate_id(), sub_graph_3)


def test_multiple_subgraphs_for_same_op():
    """Adding 2 sub graphs for the same operation (they should be merged)"""
    graph = _gen_simple_graph()

    # add a sub graph corresponding to an operation in the main graph
    node_1 = graph.get_nodes()[0]
    op_id = node_1.operation_id
    sub_graph_1 = _gen_simple_graph()
    graph.add_sub_graph(op_id, sub_graph_1)
    assert graph.get_sub_graph(op_id).get_nodes() == sub_graph_1.get_nodes()

    # add another subgraph for the same operation
    sub_graph_2 = _gen_simple_graph()
    graph.add_sub_graph(op_id, sub_graph_2)
    # both sub graphs should be merged into a new sub graph with all nodes
    merged_sub_graph = graph.get_sub_graph(op_id)
    assert merged_sub_graph.get_nodes() == sub_graph_1.get_nodes() + sub_graph_2.get_nodes()


def test_sanity_check():
    """Sanity check of graphs"""
    node_1 = ProvNode(
        data_item_id=generate_id(),
        operation_id=generate_id(),
        source_ids=[],
        derived_ids=[],
    )
    node_2 = ProvNode(
        data_item_id=generate_id(),
        operation_id=generate_id(),
        source_ids=[],
        derived_ids=[],
    )
    node_1.source_ids.append(node_2.data_item_id)
    node_2.derived_ids.append(node_1.data_item_id)
    # valid graph should not raise
    graph_1 = ProvGraph([node_1, node_2])
    graph_1.check_sanity()

    # node with sources but no operation
    node_3 = ProvNode(
        data_item_id=generate_id(),
        operation_id=None,
        source_ids=[node_2.data_item_id],
        derived_ids=[],
    )
    graph_2 = ProvGraph([node_3])
    with pytest.raises(Exception, match="Node with identifier .* has source ids but no operation"):
        graph_2.check_sanity()

    # node with source uid not corresponding to any node
    node_4 = ProvNode(
        data_item_id=generate_id(),
        operation_id=generate_id(),
        source_ids=[generate_id()],
        derived_ids=[],
    )
    graph_3 = ProvGraph([node_4])
    with pytest.raises(
        Exception,
        match=("Source identifier .* in node with identifier .* has no corresponding node"),
    ):
        graph_3.check_sanity()

    # node with derived uid not corresponding to any node
    node_4 = ProvNode(
        data_item_id=generate_id(),
        operation_id=None,
        source_ids=[],
        derived_ids=[generate_id()],
    )
    graph_4 = ProvGraph([node_4])
    with pytest.raises(
        Exception,
        match=("Derived identifier .* in node with identifier .* has no corresponding node"),
    ):
        graph_4.check_sanity()

    source_id = generate_id()
    derived_id = generate_id()
    # derived node with source uid but source node does not have derived uid
    node_5 = ProvNode(
        data_item_id=source_id,
        operation_id=None,
        source_ids=[],
        derived_ids=[],
    )
    node_6 = ProvNode(
        data_item_id=derived_id,
        operation_id=generate_id(),
        source_ids=[source_id],
        derived_ids=[],
    )
    graph_5 = ProvGraph([node_5, node_6])
    with pytest.raises(
        Exception,
        match=(
            "Node with identifier .* has source item with identifier .* but reciprocate"
            " derivation link does not exists"
        ),
    ):
        graph_5.check_sanity()

    # source node with derived uid but derived node does not have source uid
    node_5 = ProvNode(
        data_item_id=source_id,
        operation_id=None,
        source_ids=[],
        derived_ids=[derived_id],
    )
    node_6 = ProvNode(
        data_item_id=derived_id,
        operation_id=generate_id(),
        source_ids=[],
        derived_ids=[],
    )
    graph_5 = ProvGraph([node_5, node_6])
    with pytest.raises(
        Exception,
        match=(
            "Node with identifier .* has derived item with identifier .* but reciprocate source link does not exists"
        ),
    ):
        graph_5.check_sanity()

    # valid graph with invalid sub_graph
    sub_graphs_by_op_id = {generate_id(): graph_3}
    graph_6 = ProvGraph([node_1, node_2], sub_graphs_by_op_id)
    with pytest.raises(Exception, match="Source identifier .* has no corresponding node"):
        graph_6.check_sanity()
