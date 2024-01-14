from __future__ import annotations

__all__ = ["filter_overlapping_entities"]

from medkit.core.text import Entity, span_utils


def filter_overlapping_entities(entities: list[Entity]) -> list[Entity]:
    """Filter a list of entities and remove overlaps. This method may be
    useful for the creation of data for named entity recognition, where
    a part of text can only contain one entity per 'word'.
    When an overlap is detected, the longest entity is preferred.

    Parameters
    ----------
    entities : list of Entity
        Entities to filter

    Returns
    -------
    list of Entity
        Filtered entities
    """
    # concat normalize spans and entities to keep the relation after sorting
    spans_data = [(span_utils.normalize_spans(ent.spans), ent) for ent in entities]
    # sort by length and start of normalized spans, descending order
    # the longest is preferred
    sorted_spans = sorted(
        spans_data,
        key=lambda span: (
            span[0][-1].end - span[0][0].start,
            span[0][0].start,
        ),
        reverse=True,
    )
    seen_chars = set()
    filtered_entities = []
    for span, ent in sorted_spans:
        span_start = span[0].start
        span_end = span[-1].end
        if span_start not in seen_chars and span_end not in seen_chars:
            seen_chars.update(range(span_start, span_end))
            filtered_entities.append(ent)
    return sorted(filtered_entities, key=lambda ent: ent.spans[0].start)
