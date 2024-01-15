from __future__ import annotations

import logging

from medkit.core.text import Entity, Relation, Segment, TextAnnotation, TextDocument

logger = logging.getLogger(__name__)


def get_anns_by_type(medkit_doc: TextDocument, anns_labels: list[str] | None = None) -> dict[str, TextAnnotation]:
    """Filter annotations by labels and return a dictionary by type of annotation.

    Parameters
    ----------
    medkit_doc : TextDocument
        Text document with annotations
    anns_labels : list of str, optional
        Labels to filter annotations.
        If not provided, all annotations will be in the dictionary

    Returns
    -------
    Dict[str, TextAnnotation]
        Annotations by type: 'entities', 'relations', and 'segments'.

    """
    anns_by_type = {"entities": [], "relations": [], "segments": []}
    annotations = medkit_doc.anns.get()

    if anns_labels is not None:
        # filter annotations by label
        annotations = [ann for ann in annotations if ann.label in anns_labels]
        if anns_labels and annotations == []:
            # labels_anns were a list but none of the annotations
            # had a label of interest
            labels_str = ",".join(anns_labels)
            logger.info("No medkit annotations were included because none have '%s' as label.", labels_str)

    for ann in annotations:
        if isinstance(ann, Entity):
            anns_by_type["entities"].append(ann)
        elif isinstance(ann, Relation):
            anns_by_type["relations"].append(ann)
        elif isinstance(ann, Segment):
            anns_by_type["segments"].append(ann)
    return anns_by_type
