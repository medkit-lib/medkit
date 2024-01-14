from __future__ import annotations

__all__ = ["save_prov_to_dot"]

from pathlib import Path
from typing import Any, Callable, TextIO

from medkit.core import (
    Attribute,
    IdentifiableDataItem,
    IdentifiableDataItemWithAttrs,
    OperationDescription,
    Prov,
    ProvTracer,
)
from medkit.core.text import Segment, TextDocument


def save_prov_to_dot(
    prov_tracer: ProvTracer,
    file: str | Path,
    data_item_formatters: dict[type, Callable[[Any], str]] | None = None,
    op_formatter: Callable[[OperationDescription], str] | None = None,
    max_sub_prov_depth: int | None = None,
    show_attr_links: bool = True,
):
    """Generate a graphviz-compatible .dot file from a
    :class:`~medkit.core.ProvTracer` for visualization.

    Parameters
    ----------
    prov_tracer : ProvTracer
        Provenance tracer holding the provenance information to save.
    file : str or Path
        Path to the .dot file.
    data_item_formatters : dict of type to Callable, optional
        Dict mapping data items types with callback functions returning the text
        to display for each data item of this type.
        Some default formatters are available for common data item such as text
        annotations. Use `data_item_formatters` to override them or to add support
        for other types.
    op_formatter : Callable, optional
        Callback function returning the text to display for each operation.
        If none provided, the operation name is used.
    max_sub_prov_depth : int, optional
        When there are nested provenance tracers for some operations, how
        deep should we go when displaying their contents.
    show_attr_links : bool, default=True
        Whether to show links between attributes and the data items they are
        attached to (not strictly provenance but can make things easier to
        understand).
    """
    with Path(file).open(mode="w") as fp:
        writer = _DotWriter(
            fp,
            data_item_formatters,
            op_formatter,
            max_sub_prov_depth,
            show_attr_links,
        )
        writer.write_provs(prov_tracer)


_DEFAULT_DATA_ITEMS_FORMATTERS = {
    Segment: lambda s: f"{s.label}: {s.text}",
    Attribute: lambda a: f"{a.label}: {a.value}",
    TextDocument: lambda d: f"doc_text:{d.text}",
}


class _DotWriter:
    def __init__(
        self,
        fp: TextIO,
        data_item_formatters: dict[type, Callable[[Any], str]] | None,
        op_formatter: Callable[[OperationDescription], str] | None,
        max_sub_prov_depth: int | None,
        show_attr_links: bool = True,
    ):
        if data_item_formatters is None:
            data_item_formatters = {}

        self._fp = fp
        self._data_item_formatters = data_item_formatters
        self._op_formatter = op_formatter
        self._max_sub_prov_depth = max_sub_prov_depth
        self._show_attr_links = show_attr_links

    def write_provs(self, tracer: ProvTracer, current_sub_prov_depth: int = 0):
        if current_sub_prov_depth == 0:
            self._fp.write("digraph {\n\n")

        write_sub_prov = self._max_sub_prov_depth is None or current_sub_prov_depth < self._max_sub_prov_depth

        for prov in tracer.get_provs():
            if not write_sub_prov or prov.op_desc is None or not tracer.has_sub_prov_tracer(prov.op_desc.uid):
                self._write_prov(prov)

        if write_sub_prov:
            for sub_prov_tracer in tracer.get_sub_prov_tracers():
                self.write_provs(sub_prov_tracer, current_sub_prov_depth + 1)

        if current_sub_prov_depth == 0:
            self._fp.write("\n\n}")

    def _write_prov(self, prov: Prov):
        data_item = prov.data_item
        data_item_label = self._format_data_item(data_item)
        data_item_label = self._escape_quotes(data_item_label)
        self._fp.write(f'"{data_item.uid}" [label="{data_item_label}"];\n')

        if prov.op_desc is not None:
            op_label = self._op_formatter(prov.op_desc) if self._op_formatter is not None else prov.op_desc.name
            op_label = self._escape_quotes(op_label)
        else:
            op_label = "Unknown"
        for source_data_item in prov.source_data_items:
            self._fp.write(f'"{source_data_item.uid}" -> "{data_item.uid}" [label="{op_label}"];\n')
        self._fp.write("\n\n")

        if self._show_attr_links and isinstance(data_item, IdentifiableDataItemWithAttrs):
            for attr in data_item.attrs:
                self._fp.write(
                    f'"{data_item.uid}" -> "{attr.uid}" [style=dashed, color=grey,' ' label="attr", fontcolor=grey];\n'
                )

    def _format_data_item(self, data_item: IdentifiableDataItem):
        # must test first user-provided formatters then default
        # (can't merge the two in same dict, this might create unexpected
        # behavior when there a subtypes)
        for formatters in [self._data_item_formatters, _DEFAULT_DATA_ITEMS_FORMATTERS]:
            for type_, formatter in formatters.items():
                if isinstance(data_item, type_):
                    return formatter(data_item)
        msg = f"Found no formatter for data item with type {type_(data_item)}, please provide one"
        raise ValueError(msg)

    @staticmethod
    def _escape_quotes(text):
        return text.replace('"', "\\\r")
