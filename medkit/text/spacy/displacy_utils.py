__all__ = ["medkit_doc_to_displacy", "entities_to_displacy"]

from typing import Any, Callable, Dict, List, Optional

from medkit.core.text import Entity, TextDocument, span_utils


def medkit_doc_to_displacy(
    medkit_doc: TextDocument,
    entity_labels: Optional[List[str]] = None,
    entity_formatter: Optional[Callable[[Entity], str]] = None,
    max_gap_length: int = 3,
) -> Dict[str, Any]:
    """Build data dict that can be passed to `displacy.render()`
    (with `manual=True` and `style="ent"`) to visualize entities of
    a document.

    Parameters
    ----------
    medkit_doc:
        Document to visualize.
    entity_labels:
        Labels of entities to display. If `None`, all entities are displayed.
    entity_formatter:
        Optional function returning the text to display as label for a given
        entity. If `None`, the entity label will be used. Can be used for
        instance to display normalization information available in entity
        attributes.
    max_gap_length:
        When cleaning up gaps in spans, spans around gaps smaller than `max_gap_length`
        will be merged.
        Cf :func:`~medkit.core.text.span_utils.clean_up_gaps_in_normalized_spans()`.

    Returns
    -------
    Dict[str, Any]
        Data to be passed to `displacy.render()` as `docs` argument
        (with `manual=True` and `style="ent"`)
    """

    if entity_labels:
        entities = [e for label in entity_labels for e in medkit_doc.anns.get_entities(label=label)]
    else:
        entities = medkit_doc.anns.get_entities()

    return entities_to_displacy(entities, medkit_doc.text, entity_formatter, max_gap_length)


def entities_to_displacy(
    entities: List[Entity],
    raw_text: str,
    entity_formatter: Optional[Callable[[Entity], str]] = None,
    max_gap_length: int = 3,
) -> Dict[str, Any]:
    """Build data dict that can be passed to `displacy.render()`
    (with `manual=True` and `style="ent"`) to visualize entities.

    Parameters
    ----------
    entities:
        Entities to visualize in text context.
    raw_text:
        Initial document text from which entities where extracted and to which they spans refer
        (typically the `text` attribute of a :class:`~medkit.core.text.document.TextDocument`).
    entity_formatter:
        Optional function returning the text to display as label for a given
        entity. If `None`, the entity label will be used. Can be used for
        instance to display normalization information available in entity
        attributes.
    max_gap_length:
        When cleaning up gaps in spans, spans around gaps smaller than `max_gap_length`
        will be merged.
        Cf :func:`~medkit.core.text.span_utils.clean_up_gaps_in_normalized_spans()`.

    Returns
    -------
    Dict[str, Any]
        Data to be passed to `displacy.render()` as `docs` argument
        (with `manual=True` and `style="ent"`)
    """
    ents_data = []

    for entity in entities:
        normalized_spans = span_utils.normalize_spans(entity.spans)
        # normalized spans can be empty if spans contained ModifiedSpan with no replaced_spans
        if not normalized_spans:
            continue

        # merge close spans
        cleaned_spans = span_utils.clean_up_gaps_in_normalized_spans(
            normalized_spans, raw_text, max_gap_length=max_gap_length
        )

        # generate text label
        if entity_formatter:
            label = entity_formatter(entity)
        else:
            label = entity.label

        ents_data += [{"start": span.start, "end": span.end, "label": label} for span in cleaned_spans]

    ents_data = sorted(ents_data, key=lambda d: d["start"])
    return {"text": raw_text, "ents": ents_data}
