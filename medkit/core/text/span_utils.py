from __future__ import annotations

__all__ = [
    "replace",
    "remove",
    "extract",
    "insert",
    "move",
    "normalize_spans",
    "concatenate",
    "clean_up_gaps_in_normalized_spans",
]

from medkit.core.text.span import AnySpan, ModifiedSpan, Span


def _spans_have_same_length_as_text(text, spans):
    return len(text) == sum(sp.length for sp in spans)


def _lists_have_same_dimension(list_1, list_2):
    return len(list_1) == len(list_2)


def _list_is_sorted(list_1):
    return all(e <= next_e for e, next_e in zip(list_1, list_1[1:]))


def _ranges_are_within_text(text, ranges):
    return all((start <= len(text) and end <= len(text)) for start, end in ranges)


def _positions_are_within_text(text, positions):
    return all(position <= len(text) for position in positions)


def replace(
    text: str,
    spans: list[AnySpan],
    ranges: list[tuple[int, int]],
    replacement_texts: list[str],
) -> tuple[str, list[AnySpan]]:
    """Replace parts of a text, and update accordingly its associated spans

    Parameters
    ----------
    text : str
        The text in which some parts will be replaced
    spans : list of AnySpan
        The spans associated with `text`
    ranges : list of tuple of int
        The ranges of the parts that will be replaced (end excluded),
        sorted by ascending order
    replacement_texts : tuple
        The strings to use as replacements
        (must be the same length as `ranges`)

    Returns
    -------
    text : str
        The updated text
    spans : list of AnySpan
        The spans associated with the updated text

    Examples
    --------
    >>> text = "Hello, my name is John Doe."
    >>> spans = [Span(0, len(text))]
    >>> ranges = [(0, 5), (18, 22)]
    >>> replacements = ["Hi", "Jane"]
    >>> text, spans = replace(text, spans, ranges, replacements)
    >>> print(text)
    Hi, my name is Jane Doe.
    """
    # validate params
    assert _spans_have_same_length_as_text(text, spans), "Total span length should be equal to text length"
    assert _lists_have_same_dimension(
        ranges, replacement_texts
    ), "Ranges and replacement_texts should have the same dimension"
    assert _ranges_are_within_text(text, ranges), "Ranges should be within of text"
    assert _list_is_sorted(ranges), "Ranges should be sorted"

    if len(ranges) == 0:
        return text, spans

    offset = 0
    replacement_lengths = []
    for (range_start, range_end), rep_text in zip(ranges, replacement_texts):
        new_start = range_start + offset
        new_end = range_end + offset
        text = text[:new_start] + rep_text + text[new_end:]

        rep_length = len(rep_text)
        offset += rep_length - (new_end - new_start)
        replacement_lengths.append(rep_length)

    spans = _replace_in_spans(spans, ranges, replacement_lengths)
    return text, spans


def _replace_in_spans(spans, ranges, replacement_lengths):
    output_spans = []

    # current span and associated values
    span_index = 0
    span = spans[0]
    # start and end in "relative" coords (can be compared to range start/end)
    span_start = 0
    span_end = span.length

    # current range to replace and associated values
    range_index = 0
    range_start, range_end = ranges[0]
    replacement_length = replacement_lengths[0]
    replaced_spans = [] if replacement_length > 0 else None

    while span_index < len(spans) or range_index < len(ranges):
        # iterate to next range if current range has been fully handled
        if range_index < len(ranges) and range_end <= span_start:
            # we have encountered all spans overlapping with the range to replace,
            # and we have stored the overlapping parts in replaced_spans.
            # create new ModifiedSpan referencing all the replaced_spans
            # and add it to output
            # (expect if replacement_length is 0, in which case the spans were
            # just removed)
            if replacement_length > 0:
                new_span = ModifiedSpan(replacement_length, replaced_spans)
                output_spans.append(new_span)

            # move on to next span if we haven't reached end of ranges
            # and updated associated values
            range_index += 1
            if range_index < len(ranges):
                range_start, range_end = ranges[range_index]
                replacement_length = replacement_lengths[range_index]
                replaced_spans = [] if replacement_length > 0 else None

        # iterate to next span if current span has been fully handled
        if span_end == span_start or range_index == len(ranges) or span_end <= range_start:
            # add current span to output
            if span_end != span_start:
                output_spans.append(span)
            # move on to next span if we haven't reached end of spans
            # and updated associated values
            span_index += 1
            span_start = span_end  # end of previous span is start of new span
            if span_index < len(spans):
                span = spans[span_index]
                span_end = span_start + span.length
            continue

        # compute parts of span that do not overlap with current range
        length_before_range = max(range_start - span_start, 0)
        length_after_range = max(span_end - range_end, 0)

        # store part of span that will be replaced
        if replacement_length > 0 and length_before_range + length_after_range < span.length:
            if isinstance(span, Span):
                replaced_span = Span(
                    start=span.start + length_before_range,
                    end=span.end - length_after_range,
                )
                replaced_spans.append(replaced_span)
            else:
                # keep reference to all the replaced_spans in original ModifiedSpan
                # (not possible to know which subpart of the replaced_spans corresponds
                # to the overlap between the ModifiedSpan and the range)
                assert isinstance(span, ModifiedSpan)
                replaced_spans += span.replaced_spans

        # create span for the part before the range
        # and add it to output
        if length_before_range > 0:
            if isinstance(span, Span):
                before_span = Span(start=span.start, end=span.start + length_before_range)
            else:
                # create new ModifiedSpan covering only the length before the range,
                # but referencing the same replaced_spans
                # (not possible to know which subpart of the replaced_spans corresponds
                # to the part of the ModifiedSpan before the range)
                assert isinstance(span, ModifiedSpan)
                before_span = ModifiedSpan(length=length_before_range, replaced_spans=span.replaced_spans)
            output_spans.append(before_span)

        # create span for the remaining part after the range
        # and use it as current span
        if length_after_range > 0:
            if isinstance(span, Span):
                span = Span(start=span.end - length_after_range, end=span.end)
            else:
                # create new ModifiedSpan covering only the length after the range,
                # but referencing the same replaced_spans
                # (not possible to know which subpart of the replaced_spans corresponds
                # to the part of the ModifiedSpan after the range)
                assert isinstance(span, ModifiedSpan)
                span = ModifiedSpan(length=length_after_range, replaced_spans=span.replaced_spans)
        # update span_start to point to the beginning of the remainder
        span_start = span_end - length_after_range

    return output_spans


def remove(
    text: str,
    spans: list[AnySpan],
    ranges: list[tuple[int, int]],
) -> tuple[str, list[AnySpan]]:
    """Remove parts of a text, while also removing accordingly its associated spans

    Parameters
    ----------
    text : str
        The text in which some parts will be removed
    spans : list of AnySpan
        The spans associated with `text`
    ranges : list of tuple of int
        The ranges of the parts that will be removed (end excluded),
        sorted by ascending order

    Returns
    -------
    text : str
        The updated text
    spans : list of AnySpan
        The spans associated with the updated text
    """
    # validate params
    assert _spans_have_same_length_as_text(text, spans), "Total span length should be equal to text length"
    assert _ranges_are_within_text(text, ranges), "Ranges should be within of text"
    assert _list_is_sorted(ranges), "Ranges should be sorted"

    if len(ranges) == 0:
        return text, spans

    offset = 0
    for range_start, range_end in ranges:
        new_start = range_start + offset
        new_end = range_end + offset
        text = text[:new_start] + text[new_end:]
        offset -= new_end - new_start
    spans = _remove_in_spans(spans, ranges)
    return text, spans


def _remove_in_spans(spans, ranges):
    replacement_lengths = [0] * len(ranges)
    return _replace_in_spans(spans, ranges, replacement_lengths)


def extract(
    text: str,
    spans: list[AnySpan],
    ranges: list[tuple[int, int]],
) -> tuple[str, list[AnySpan]]:
    """Extract parts of a text as well as its associated spans

    Parameters
    ----------
    text : str
        The text to extract parts from
    spans : list of AnySpan
        The spans associated with `text`
    ranges : list of tuple of int
        The ranges of the parts to extract (end excluded),
        sorted by ascending order

    Returns
    -------
    text : str
        The extracted text
    spans : list of AnySpan
        The spans associated with the extracted text
    """
    # validate params
    assert _spans_have_same_length_as_text(text, spans), "Total span length should be equal to text length"
    assert _ranges_are_within_text(text, ranges), "Ranges should be within of text"
    assert _list_is_sorted(ranges), "Ranges should be sorted"

    if len(ranges) == 0:
        return "", []

    text = "".join(text[s:e] for s, e in ranges)
    spans = _extract_in_spans(spans, ranges)
    return text, spans


def _extract_in_spans(spans, ranges):
    ranges_to_remove = []

    first_range_start = ranges[0][0]
    ranges_to_remove.append((0, first_range_start))

    ranges_to_remove += [(end_1, start_2) for (_, end_1), (start_2, _) in zip(ranges, ranges[1:])]

    last_range_end = ranges[-1][1]
    total_length = sum(s.length for s in spans)
    ranges_to_remove.append((last_range_end, total_length))
    return _remove_in_spans(spans, ranges_to_remove)


def insert(
    text: str,
    spans: list[AnySpan],
    positions: list[int],
    insertion_texts: list[str],
) -> tuple[str, list[AnySpan]]:
    """Insert strings in text, and update accordingly its associated spans

    Parameters
    ----------
    text : str
        The text in which some strings will be inserted
    spans : list of AnySpan
        The spans associated with `text`
    positions : list of int
        The positions where the strings will be inserted,
        sorted by ascending order
    insertion_texts : list of str
        The strings to insert (must be the same length as `positions`)

    Returns
    -------
    text : str
        The updated text
    spans : list of AnySpan
        The spans associated with the updated text

    Examples
    --------
    >>> text = "Hello, my name is John Doe."
    >>> spans = [Span(0, len(text))]
    >>> positions = [5]
    >>> inserts = [" everybody"]
    >>> text, spans = insert(text, spans, positions, inserts)
    >>> print(text)
    Hello everybody, my name is John Doe."
    """
    # validate params
    assert _spans_have_same_length_as_text(text, spans), "Total span length should be equal to text length"
    assert _lists_have_same_dimension(
        positions, insertion_texts
    ), "Positions and insertion_texts should have the same dimension"
    assert _positions_are_within_text(text, positions)
    assert _list_is_sorted(positions), "Positions should be sorted"

    if len(positions) == 0:
        return text, spans

    offset = 0
    insertion_lengths = []
    for position, insertion_text in zip(positions, insertion_texts):
        new_position = position + offset
        text = text[:new_position] + insertion_text + text[new_position:]

        insertion_length = len(insertion_text)
        offset += insertion_length
        insertion_lengths.append(insertion_length)

    spans = _insert_in_spans(spans, positions, insertion_lengths)
    return text, spans


def _insert_in_spans(spans, positions, insertion_lengths):
    # build zero-length ranges for each insertion position
    # (end is not included)
    # we "replace" a zero-length range with the inserted text,
    # so no text actually ends up being replaced
    ranges = [(p, p) for p in positions]
    return _replace_in_spans(spans, ranges, insertion_lengths)


def move(
    text: str,
    spans: list[AnySpan],
    range: tuple[int, int],  # noqa: A002
    destination: int,
) -> tuple[str, list[AnySpan]]:
    """Move part of a text to another position, also moving its associated spans

    Parameters
    ----------
    text : str
        The text in which a part should be moved
    spans : list of AnySpan
        The spans associated with the input text
    range : tuple of int
        The range of the part to move (end excluded)
    destination : int
        The position where to insert the displaced range

    Returns
    -------
    text : str
        The updated text
    spans : list of AnySpan
        The spans associated with the updated text

    Examples
    --------
    >>> text = "Hello, my name is John Doe."
    >>> spans = [Span(0, len(text))]
    >>> range = (17, 22)
    >>> dest = len(text) - 1
    >>> text, spans = move(text, spans, range, dest)
    >>> print(text)
    Hi, my name is Doe John.
    """
    spans = _move_in_spans(spans, range, destination)

    start, end = range
    text_in_moved_range = text[start:end]
    text_without_moved_range = text[:start] + text[end:]
    # shift destination if it was after the moved range
    if destination > end:
        length = end - start
        destination -= length
    text_before_dest = text_without_moved_range[:destination]
    text_after_dest = text_without_moved_range[destination:]
    text = text_before_dest + text_in_moved_range + text_after_dest

    return text, spans


def _move_in_spans(spans, range, destination):  # noqa: A002
    start, end = range
    length = end - start
    assert not (start < destination <= end)
    spans_to_move = _extract_in_spans(spans, [(start, end)])

    spans = _remove_in_spans(spans, [(start, end)])
    if destination > end:
        destination -= length

    spans_before = _extract_in_spans(spans, [(0, destination)]) if destination > 0 else []
    total_length = sum(s.length for s in spans)
    spans_after = _extract_in_spans(spans, [(destination, total_length)]) if destination < total_length else []

    return spans_before + spans_to_move + spans_after


def concatenate(texts: list[str], all_spans: list[list[AnySpan]]) -> tuple[str, list[AnySpan]]:
    """Concatenate text and span objects"""
    assert _lists_have_same_dimension(texts, all_spans), "Text and all_spans should have the same dimension"
    text = "".join(texts)
    span = [sp for spans in all_spans for sp in spans]

    return text, span


def normalize_spans(spans: list[AnySpan]) -> list[Span]:
    """Return a transformed of `spans` in which all instances of ModifiedSpan are
    replaced by the spans they refer to, spans are sorted and contiguous spans are merged.

    Parameters
    ----------
    spans : list of AnySpan
        The spans associated with a text, including additional spans if
        insertions or replacement were performed

    Returns
    -------
    normalized_spans : list of Span
        Spans in `spans` normalized as described

    Examples
    --------
    >>> spans = [
    ...     Span(0, 10),
    ...     Span(20, 30),
    ...     ModifiedSpan(8, replaced_spans=[Span(30, 36)]),
    ... ]
    >>> spans = normalize_spans(spans)
    >>> print(spans)
    >>> [Span(0, 10), Span(20, 36)]
    """
    all_spans = []
    for span in spans:
        if isinstance(span, ModifiedSpan):
            all_spans += span.replaced_spans
        else:
            assert isinstance(span, Span)
            all_spans.append(span)

    if not all_spans:
        return []

    all_spans.sort(key=lambda s: s.start)
    # merge contiguous spans
    all_spans_merged = [all_spans[0]]
    for span in all_spans[1:]:
        prev_span = all_spans_merged[-1]
        if span.start == prev_span.end:
            merged_span = Span(prev_span.start, span.end)
            all_spans_merged[-1] = merged_span
        else:
            all_spans_merged.append(span)

    return all_spans_merged


def clean_up_gaps_in_normalized_spans(spans: list[Span], text: str, max_gap_length: int = 3):
    """Remove small gaps in normalized spans.

    This is useful for converting non-contiguous entity spans with small gaps containing
    only whitespace or a few meaningless characters (due to clean-up preprocessing
    or translation) into one unique bigger span. Gaps having less than `max_gap_length`
    will be removed by merging the spans before and after the gap.

    Parameters
    ----------
    spans : list of Span
        The normalized spans in which to remove gaps
    text : str
        The text associated with `spans`
    max_gap_length : int, default=3
        Max number of characters in gaps, after stripping leading and trailing whitespace.

    Examples
    --------
    >>> text = "heart failure"
    >>> spans = [Span(0, 5), Span(6, 13)]
    >>> spans = clean_up_gaps_in_normalized_spans(spans, text)
    >>> print(spans)
    >>> spans = [Span(0, 13)]
    """
    spans_merged = [spans[0]]
    for span in spans[1:]:
        prev_span = spans_merged[-1]
        gap_text = text[prev_span.end : span.start]
        if len(gap_text.strip()) <= max_gap_length:
            merged_span = Span(prev_span.start, span.end)
            spans_merged[-1] = merged_span
        else:
            spans_merged.append(span)

    return spans_merged
