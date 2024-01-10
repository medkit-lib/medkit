__all__ = [
    "transform_entities_to_tags",
    "align_and_map_tokens_with_tags",
    "convert_labels_to_tags",
]

from typing import Dict, List

from transformers.tokenization_utils_fast import EncodingFast
from typing_extensions import Literal

from medkit.core.text import Entity, span_utils

SPECIAL_TAG_ID_HF: int = -100


def convert_labels_to_tags(
    labels: List[str],
    tagging_scheme: Literal["bilou", "iob2"] = "bilou",
) -> Dict[str, int]:
    """Convert a list of labels in a mapping of NER tags

    Parameters
    ----------
    labels : list of str
        List of labels to convert
    tagging_scheme : str, default="bilou"
        Scheme to use in the conversion, "iob2" follows the BIO scheme.

    Returns
    -------
    dict of str to int
        Mapping with NER tags.

    Examples
    --------
    >>> convert_labels_to_tags(labels=["test", "problem"], tagging_scheme="iob2")
    {'O': 0, 'B-test': 1, 'I-test': 2, 'B-problem': 3, 'I-problem': 4}

    """
    label_to_id = {}
    label_to_id["O"] = 0

    scheme = ["B", "I", "L", "U"] if tagging_scheme == "bilou" else ["B", "I"]

    all_labels = [f"{prefix}-{label}" for label in labels for prefix in scheme]

    for idx, label in enumerate(all_labels):
        label_to_id[label] = idx + 1
    return label_to_id


def create_entity_tags(nb_tags: int, label: str, tagging_scheme: Literal["bilou", "iob2"]) -> List[str]:
    """Create a list of tags representing one entity

    Parameters
    ----------
    nb_tags:
        Total of tags to create
    label:
        Entity label
    tagging_scheme:
        Scheme to use in the conversion, "iob2" follows the BIO scheme.

    Returns
    -------
    tags: List[str]:
        Tags representing the entity

    Examples
    --------
    >>> create_entity_tags(nb_tags=3, label="corporation", tagging_scheme="bilou")
    ['B-corporation', 'I-corporation', 'L-corporation']
    """
    tags = [f"I-{label}"] * nb_tags
    if len(tags) == 1:
        prefix = "U" if tagging_scheme == "bilou" else "B"
        tags[0] = f"{prefix}-{label}"
    else:
        tags[0] = f"B-{label}"
        prefix = "L" if tagging_scheme == "bilou" else "I"
        tags[-1] = f"{prefix}-{label}"
    return tags


def transform_entities_to_tags(
    text_encoding: EncodingFast,
    entities: List[Entity],
    tagging_scheme: Literal["bilou", "iob2"] = "bilou",
) -> List[str]:
    """Transform entities from a encoded document to a list of BILOU/IOB2 tags.

    Parameters
    ----------
    text_encoding:
        Encoding of the document of reference, this is created by a HuggingFace fast tokenizer.
        It contains a tokenized version of the document to tag.
    entities:
        The list of entities to transform
    tagging_scheme:
        Scheme to tag the tokens, it can be `bilou` or `iob2`

    Returns
    -------
    List[str]:
        A list describing the document with tags. By default the tags
        could be "B", "I", "L", "O","U", if `tagging_scheme` is `iob2`
        the tags could be "B", "I","O".

    Examples
    --------
    >>> # Define a fast tokenizer, i.e. : bert tokenizer
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    >>> document = TextDocument(text="medkit")
    >>> entities = [
    ...     Entity(label="corporation", spans=[Span(start=0, end=6)], text="medkit")
    ... ]
    >>> # Get text encoding of the document using the tokenizer
    >>> text_encoding = tokenizer(document.text).encodings[0]
    >>> print(text_encoding.tokens)
    ['[CLS]', 'med',##kit', '[SEP]']

    Transform to BILOU tags

    >>> tags = transform_entities_to_tags(text_encoding, entities)
    >>> assert tags == ["O", "B-corporation", "L-corporation", "O"]

    Transform to IOB2 tags

    >>> tags = transform_entities_to_tags(text_encoding, entities, "iob2")
    >>> assert tags == ["O", "B-corporation", "I-corporation", "O"]


    """
    tags = ["O"] * len(text_encoding)

    for ent in entities:
        label = ent.label
        ent_spans = span_utils.normalize_spans(ent.spans)
        start_char = ent_spans[0].start
        end_char = ent_spans[-1].end
        tokens_entity = set()

        for idx in range(start_char, end_char):
            token_id = text_encoding.char_to_token(idx)

            if token_id is not None:
                tokens_entity.add(token_id)

        tokens_entity = sorted(tokens_entity)

        if not tokens_entity:
            continue

        entity_tags = create_entity_tags(nb_tags=len(tokens_entity), label=label, tagging_scheme=tagging_scheme)

        for token_idx, tag in zip(tokens_entity, entity_tags):
            tags[token_idx] = tag
    return tags


def align_and_map_tokens_with_tags(
    text_encoding: EncodingFast,
    tags: List[str],
    tag_to_id: Dict[str, int],
    map_sub_tokens: bool = True,
) -> List[int]:
    """Return a list of tags_ids aligned with the text encoding.
    Tags considered as special tokens will have the `SPECIAL_TAG_ID_HF`.

    Parameters
    ----------
    text_encoding:
        Text encoding after tokenization with a HuggingFace fast tokenizer
    tags:
        A list of tags i.e BILOU tags
    tag_to_id:
        Mapping tag to id
    map_sub_tokens:
        When a token is not in the vocabulary of the tokenizer, it could split
        the token into multiple subtokens.
        If `map_sub_tokens` is True, all tags inside a token will be converted.
        If `map_sub_tokens` is False, only the first subtoken of a split token will be
        converted.

    Returns
    -------
    List[int]:
        A list of tags ids

    Examples
    --------
    >>> # Define a fast tokenizer, i.e. : bert tokenizer
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    >>> # define data to map
    >>> text_encoding = tokenizer("medkit").encodings[0]
    >>> tags = ["O", "B-corporation", "I-corporation", "O"]
    >>> tag_to_id = {"O": 0, "B-corporation": 1, "I-corporation": 2}
    >>> print(text_encoding.tokens)
    ['[CLS]', 'med',##kit', '[SEP]']

    Mapping all tags to tags_ids

    >>> tags_ids = align_and_map_tokens_with_tags(text_encoding, tags, tag_to_id)
    >>> assert tags_ids == [-100, 1, 2, -100]

    Mapping only first tag in tokens

    >>> tags_ids = align_and_map_tokens_with_tags(text_encoding, tags, tag_to_id, False)
    >>> assert tags_ids == [-100, 1, -100, -100]
    """
    special_tokens_mask = text_encoding.special_tokens_mask

    tags_ids = [SPECIAL_TAG_ID_HF] * len(tags)
    words = text_encoding.word_ids

    prev_word = None
    for idx, label in enumerate(tags):
        if special_tokens_mask[idx]:
            continue

        current_word = words[idx]
        if current_word != prev_word:
            # map the first token of the word
            tags_ids[idx] = tag_to_id[label]
            prev_word = current_word

        if map_sub_tokens:
            tags_ids[idx] = tag_to_id[label]
    return tags_ids
