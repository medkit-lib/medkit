from __future__ import annotations

__all__ = ["compute_nested_segments"]

from intervaltree import IntervalTree

from medkit.core.text import Segment, span_utils


def _create_segments_tree(target_segments: list[Segment]) -> IntervalTree:
    """Use the normalized spans of the segments to create an interval tree

    Parameters
    ----------
    target_segments : list of Segment
        List of segments to align

    Returns
    -------
    IntervalTree
        Interval tree from the target segments
    """
    tree = IntervalTree()
    for segment in target_segments:
        normalized_spans = span_utils.normalize_spans(segment.spans)

        if not normalized_spans:
            continue

        tree.addi(
            normalized_spans[0].start,
            normalized_spans[-1].end,
            data=segment,
        )
    return tree


def compute_nested_segments(
    source_segments: list[Segment],
    target_segments: list[Segment],
) -> list[tuple[Segment, list[Segment]]]:
    """Return source segments aligned with its nested segments.
    Only nested segments fully contained in the `source_segments` are returned.

    Parameters
    ----------
    source_segments : list of Segment
        List of source segments
    target_segments : list of Segment
        List of segments to align

    Returns
    -------
    list of tuple
        List of aligned segments
    """
    tree = _create_segments_tree(target_segments)
    nested = []
    for parent in source_segments:
        normalized_spans = span_utils.normalize_spans(parent.spans)

        if not normalized_spans:
            continue
        start, end = normalized_spans[0].start, normalized_spans[-1].end
        # use 'tree.envelop' to get only fully contained children
        children = [child.data for child in tree.envelop(start, end)]
        nested.append((parent, children))
    return nested
