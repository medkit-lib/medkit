from __future__ import annotations

__all__ = ["ProvGraph", "ProvNode"]

import dataclasses
from typing import Any


@dataclasses.dataclass
class ProvNode:
    data_item_id: str
    operation_id: str | None
    source_ids: list[str]
    derived_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_item_id": self.data_item_id,
            "operation_id": self.operation_id,
            "source_ids": self.source_ids,
            "derived_ids": self.derived_ids,
        }


class ProvGraph:
    def __init__(
        self,
        nodes: list[ProvNode] | None = None,
        sub_graphs_by_op_id: dict[str, ProvGraph] | None = None,
    ):
        if nodes is None:
            nodes = []
        if sub_graphs_by_op_id is None:
            sub_graphs_by_op_id = {}

        self._nodes_by_id: dict[str, ProvNode] = {n.data_item_id: n for n in nodes}
        self._sub_graphs_by_op_id: dict[str, ProvGraph] = sub_graphs_by_op_id

    def get_nodes(self) -> list[ProvNode]:
        return list(self._nodes_by_id.values())

    def get_node(self, data_item_id: str) -> ProvNode:
        return self._nodes_by_id[data_item_id]

    def add_node(self, data_item_id: str, operation_id: str, source_ids: list[str]):
        """Create a node describing how a data item was created.

        Parameters
        ----------
        data_item_id: str
            Identifier of the data item that was created.
        operation_id: str
            Identifier of the operation that created the data item.
        source_ids: list of str
            Identifier of pre-existing data items from which the data item was derived
            (if any). If these source data items don't have corresponding nodes,
            "stub" nodes (nodes with no `operation_id` and no `source_ids`) are
            created. It allows us to know how a data item was used even if we
            don't know how it was created.
        """
        node = self._nodes_by_id.get(data_item_id)
        # 2 different cases may occur:
        # - there is no node for data_item_id. This is the most straightforward
        #   case, we just create a node with the provided operation_id and
        #   source_ids;
        # - there is already a node for data_item_id. Even though add_node()
        #   should only be called only once for a data item, a "stub" node (a
        #   node with no operation_id and no source_ids) may already exists if
        #   data_item_id was used as a source_id in a previous call to
        #   add_node(). In this case, we update the existing node by setting its
        #   operation_id and source_ids to the provided values.
        if node is None:
            # no node exist for the data item, we just have to create one
            node = ProvNode(
                data_item_id=data_item_id,
                operation_id=operation_id,
                source_ids=source_ids,
                derived_ids=[],
            )
            self._nodes_by_id[data_item_id] = node
        else:
            # a node already exists for the data item. this is valid only if the
            # node is a "stub" node, otherwise it probably means that add_node()
            # has been called twice with the same data_item_id
            if node.operation_id is not None:
                msg = f"Node with uid {data_item_id} already added to graph"
                raise ValueError(msg)
            # check consistency of stub node: operation_id should be None, and
            # source_ids should be empty
            if len(node.source_ids) != 0:
                msg = "Inconsistent values for stub node: operation_id is None but source_ids is not empty"
                raise ValueError(msg)
            # we are now sure that the node is a stub node and that operation_id
            # and source_ids are empty and can be set to the provided values
            node.operation_id = operation_id
            node.source_ids = source_ids

        # update derivation edges of source nodes
        for source_id in source_ids:
            source_node = self._nodes_by_id.get(source_id)
            # if source item is unknown to graph,
            # create stub node with no operation
            if source_node is None:
                source_node = ProvNode(
                    source_id,
                    operation_id=None,
                    source_ids=[],
                    derived_ids=[],
                )
                self._nodes_by_id[source_id] = source_node
            source_node.derived_ids.append(data_item_id)

    def has_node(self, data_item_id: str) -> bool:
        return data_item_id in self._nodes_by_id

    def get_sub_graphs(self) -> list[ProvGraph]:
        return list(self._sub_graphs_by_op_id.values())

    def has_sub_graph(self, operation_id: str) -> bool:
        return operation_id in self._sub_graphs_by_op_id

    def get_sub_graph(self, operation_id: str) -> ProvGraph:
        return self._sub_graphs_by_op_id[operation_id]

    def add_sub_graph(self, operation_id: str, sub_graph: ProvGraph):
        if operation_id in self._sub_graphs_by_op_id:
            current_sub_graph = self._sub_graphs_by_op_id[operation_id]
            new_sub_graph = current_sub_graph._merge(sub_graph)
            self._sub_graphs_by_op_id[operation_id] = new_sub_graph
        else:
            self._sub_graphs_by_op_id[operation_id] = sub_graph

    def _merge(self, other_graph: ProvGraph) -> ProvGraph:
        merged_prov_graph = ProvGraph()
        merged_prov_graph._nodes_by_id = {
            **self._nodes_by_id,
            **other_graph._nodes_by_id,
        }
        merged_prov_graph._sub_graphs_by_op_id = {
            **self._sub_graphs_by_op_id,
            **other_graph._sub_graphs_by_op_id,
        }
        return merged_prov_graph

    def check_sanity(self):
        for node_id, node in self._nodes_by_id.items():
            if node.source_ids and node.operation_id is None:
                msg = f"Node with identifier {node_id} has source ids but no operation"
                raise ValueError(msg)
            for source_id in node.source_ids:
                source_node = self._nodes_by_id.get(source_id)
                if source_node is None:
                    msg = f"Source identifier {source_id} in node with identifier {node_id} has no corresponding node"
                    raise ValueError(msg)
                if node_id not in source_node.derived_ids:
                    msg = (
                        f"Node with identifier {node_id} has source item with"
                        f" identifier {source_id} but reciprocate derivation link does"
                        " not exists"
                    )
                    raise ValueError(msg)
            for derived_id in node.derived_ids:
                derived_node = self._nodes_by_id.get(derived_id)
                if derived_node is None:
                    msg = (
                        f"Derived identifier {derived_id} in node with identifier"
                        f" {node_id} has no corresponding node"
                    )
                    raise ValueError(msg)
                if node_id not in derived_node.source_ids:
                    msg = (
                        f"Node with identifier {node_id} has derived item with"
                        f" identifier {derived_id} but reciprocate source link does not"
                        " exists"
                    )
                    raise ValueError(msg)
        for sub_graph in self._sub_graphs_by_op_id.values():
            sub_graph.check_sanity()

    def to_dict(self) -> dict[str, Any]:
        nodes = [n.to_dict() for n in self._nodes_by_id.values()]
        sub_graphs_by_op_id = {uid: s.to_dict() for uid, s in self._sub_graphs_by_op_id.items()}
        return {"nodes": nodes, "sub_graphs_by_op_id": sub_graphs_by_op_id}
