from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from smart_open import open

if TYPE_CHECKING:
    from pathlib import Path

GROUPING_ENTITIES = frozenset(["And-Group", "Or-Group"])
GROUPING_RELATIONS = frozenset(["And", "Or"])

logger = logging.getLogger(__name__)


@dataclass
class BratEntity:
    """A simple entity annotation data structure."""

    uid: str
    type: str
    span: list[tuple[int, int]]
    text: str

    @property
    def start(self) -> int:
        return self.span[0][0]

    @property
    def end(self) -> int:
        return self.span[-1][-1]

    def to_str(self) -> str:
        spans_str = ";".join(f"{span[0]} {span[1]}" for span in self.span)
        return f"{self.uid}\t{self.type} {spans_str}\t{self.text}\n"


@dataclass
class BratRelation:
    """A simple relation data structure."""

    uid: str
    type: str
    subj: str
    obj: str

    def to_str(self) -> str:
        return f"{self.uid}\t{self.type} Arg1:{self.subj} Arg2:{self.obj}\n"


@dataclass
class BratAttribute:
    """A simple attribute data structure."""

    uid: str
    type: str
    target: str
    value: str = None  # Only one value is possible

    def to_str(self) -> str:
        value = ensure_attr_value(self.value)
        value_str = f" {value}" if value else ""
        return f"{self.uid}\t{self.type} {self.target}{value_str}\n"


@dataclass
class BratNote:
    """A simple note data structure."""

    uid: str
    target: str
    value: str
    type: str = "AnnotatorNotes"

    def to_str(self) -> str:
        return f"{self.uid}\t{self.type} {self.target}\t{self.value}\n"


def ensure_attr_value(attr_value: Any) -> str:
    """Ensure that `attr_value` is a string. If it's not, the
    value is changed depending on its original format.
    """
    if isinstance(attr_value, str):
        return attr_value
    if attr_value is None or isinstance(attr_value, bool):
        return ""
    if isinstance(attr_value, list):
        # list is not supported in Brat
        msg = "Its value is a list and this is not supported by Brat"
        raise TypeError(msg)
    return str(attr_value)


@dataclass
class Grouping:
    """A grouping data structure for entities  of type 'And-Group", "Or-Group'"""

    uid: str
    type: str
    items: list[BratEntity]

    @property
    def text(self):
        return f" {self.type.split('-')[0]} ".join(i.text for i in self.items)


@dataclass
class BratAugmentedEntity:
    """An augmented entity data structure with its relations and attributes."""

    uid: str
    type: str
    span: tuple[tuple[int, int], ...]
    text: str
    relations_from_me: tuple[BratRelation, ...]
    relations_to_me: tuple[BratRelation, ...]
    attributes: tuple[BratAttribute, ...]

    @property
    def start(self) -> int:
        return self.span[0][0]

    @property
    def end(self) -> int:
        return self.span[-1][-1]


@dataclass
class BratDocument:
    entities: dict[str, BratEntity]
    relations: dict[str, BratRelation]
    attributes: dict[str, BratAttribute]
    notes: dict[str, BratNote]
    groups: dict[str, Grouping] = None

    def get_augmented_entities(self) -> dict[str, BratAugmentedEntity]:
        augmented_entities = {}
        for entity in self.entities.values():
            entity_relations_from_me = tuple(
                relation for relation in self.relations.values() if relation.subj == entity.uid
            )
            entity_relations_to_me = tuple(
                relation for relation in self.relations.values() if relation.obj == entity.uid
            )
            entity_attributes = tuple(
                attribute for attribute in self.attributes.values() if attribute.target == entity.uid
            )
            augmented_entities[entity.uid] = BratAugmentedEntity(
                uid=entity.uid,
                type=entity.type,
                span=entity.span,
                text=entity.text,
                relations_from_me=entity_relations_from_me,
                relations_to_me=entity_relations_to_me,
                attributes=entity_attributes,
            )
        return augmented_entities


# data structures for configuration
class RelationConf(NamedTuple):
    """Configuration data structure of a BratRelation"""

    type: str
    arg1: str
    arg2: str


class AttributeConf(NamedTuple):
    """Configuration data structure of a BratAttribure"""

    from_entity: bool
    type: str
    value: str


class BratAnnConfiguration:
    """A data structure to represent 'annotation.conf' in brat documents.
    This is necessary to generate a valid annotation project in brat.
    An 'annotation.conf' has four sections. The section 'events' is not
    supported in medkit, so the section is empty.
    """

    def __init__(self, top_values_by_attr: int = 50):
        self._entity_types: set[str] = set()
        # key: relation type
        self._rel_types_arg_1: dict[str, set[str]] = defaultdict(set)
        # key: relation type
        self._rel_types_arg_2: dict[str, set[str]] = defaultdict(set)
        # key: attribute type
        self._attr_entity_values: dict[str, list[str]] = defaultdict(list)
        self._attr_relation_values: dict[str, list[str]] = defaultdict(list)
        # 'n' most common values by attr to be included in the conf file
        self.top_values_by_attr = top_values_by_attr

    # return sorted version of BratAnnotationConfiguration
    @property
    def entity_types(self) -> list[str]:
        return sorted(self._entity_types)

    @property
    def rel_types_arg_1(self) -> dict[str, list[str]]:
        rels = {}
        for rel_type, values in self._rel_types_arg_1.items():
            rels[rel_type] = sorted(values)
        return rels

    @property
    def rel_types_arg_2(self) -> dict[str, list[str]]:
        rels = {}
        for rel_type, values in self._rel_types_arg_2.items():
            rels[rel_type] = sorted(values)
        return rels

    # as brat only allows defined values, certain data types
    # are not fully supported (e.g. int, float).
    # We limit the number of different values of an attribute
    # to show in the configuration.
    @property
    def attr_relation_values(self) -> dict[str, list[str]]:
        attrs = {}
        for attr_type, values in self._attr_relation_values.items():
            # get the 'n' most common values in the attr
            most_common_values = Counter(values).most_common(self.top_values_by_attr)
            attrs[attr_type] = sorted(attr_value for attr_value, _ in most_common_values)
        return attrs

    @property
    def attr_entity_values(self) -> dict[str, list[str]]:
        attrs = {}
        for attr_type, values in self._attr_entity_values.items():
            # get the 'n' most common values in the attr
            most_common_values = Counter(values).most_common(self.top_values_by_attr)
            attrs[attr_type] = sorted(attr_value for attr_value, _ in most_common_values)
        return attrs

    def add_entity_type(self, type: str):
        self._entity_types.add(type)

    def add_relation_type(self, relation_conf: RelationConf):
        self._rel_types_arg_1[relation_conf.type].add(relation_conf.arg1)
        self._rel_types_arg_2[relation_conf.type].add(relation_conf.arg2)

    def add_attribute_type(self, attr_conf: AttributeConf):
        if attr_conf.from_entity:
            self._attr_entity_values[attr_conf.type].append(attr_conf.value)
        else:
            self._attr_relation_values[attr_conf.type].append(attr_conf.value)

    def to_str(self) -> str:
        annotation_conf = (
            "#Text-based definitions of entity types, relation types\n"
            "#and attributes. This file was generated using medkit\n"
            "#from the HeKa project"
        )
        annotation_conf += "\n[entities]\n\n"
        entity_section = "\n".join(self.entity_types)
        annotation_conf += entity_section

        # add relations section
        annotation_conf += "\n[relations]\n\n"
        annotation_conf += "# This line enables entity overlapping\n"
        annotation_conf += "<OVERLAP>\tArg1:<ENTITY>, Arg2:<ENTITY>, <OVL-TYPE>:<ANY>\n"

        rel_types_arg_1 = self.rel_types_arg_1
        rel_types_arg_2 = self.rel_types_arg_2
        for type in rel_types_arg_1:
            arg_1_types = rel_types_arg_1[type]
            arg_2_types = rel_types_arg_2[type]
            relation_line = self._relation_to_str(type, arg_1_types, arg_2_types)
            annotation_conf += f"{relation_line}\n"

        # add attributes section
        attr_entity_values = self.attr_entity_values
        annotation_conf += "[attributes]\n\n"
        for type, values in attr_entity_values.items():
            attr_line = self._attribute_to_str(type, values, True)
            annotation_conf += f"{attr_line}\n"

        attr_relation_values = self.attr_relation_values
        for type, values in attr_relation_values.items():
            attr_line = self._attribute_to_str(type, values, False)
            annotation_conf += f"{attr_line}\n"
        # add events section (empty)
        annotation_conf += "[events]\n\n"
        return annotation_conf

    @staticmethod
    def _attribute_to_str(type: str, values: list[str], from_entity: bool) -> str:
        arg = "<ENTITY>" if from_entity else "<RELATION>"
        values_str = "|".join(values)
        return f"{type}\tArg:{arg}" if not values_str else f"{type}\tArg:{arg}, Value:{values_str}"

    @staticmethod
    def _relation_to_str(type: str, arg_1_types: list[str], arg_2_types: list[str]) -> str:
        arg_1_str = "|".join(arg_1_types)
        arg_2_str = "|".join(arg_2_types)
        return f"{type}\tArg1:{arg_1_str}, Arg2:{arg_2_str}"


def parse_file(ann_path: str | Path, detect_groups: bool = False) -> BratDocument:
    """Read an annotation file to get the Entities, Relations and Attributes in it.
    All other lines are ignored.

    Parameters
    ----------
    ann_path : str or Path
        The path to the annotation file to be processed.
    detect_groups : bool, default=False
        If set to `True`, the function will also parse the group of entities according
        to some specific keywords.
        By default, it is set to False.

    Returns
    -------
    Document
        The dataclass object containing entities, relations and attributes

    """
    with open(ann_path, encoding="utf-8") as ann_file:
        ann_content = ann_file.read()
    return parse_string(ann_content, detect_groups)


def parse_string(ann_string: str, detect_groups: bool = False) -> BratDocument:
    """Read a string containing all annotations and extract Entities, Relations and
    Attributes.
    All other lines are ignored.

    Parameters
    ----------
    ann_string : str
        The string containing all brat annotations
    detect_groups : bool, default=False
        If set to `True`, the function will also parse the group of entities according
        to some specific keywords.
        By default, it is set to False.

    Returns
    -------
    Document
        The dataclass object containing entities, relations and attributes
    """
    entities = {}
    relations = {}
    attributes = {}
    notes = {}

    annotations = ann_string.split("\n")
    for i, ann in enumerate(annotations):
        line_number = i + 1
        if len(ann) == 0 or ann[0] not in ("T", "R", "A", "#"):
            logger.info("Ignoring empty line or unsupported annotation %s on %s", ann, line_number)
            continue
        ann_id, ann_content = ann.split("\t", maxsplit=1)
        try:
            if ann.startswith("T"):
                entity = _parse_entity(ann_id, ann_content)
                entities[entity.uid] = entity
            elif ann.startswith("R"):
                relation = _parse_relation(ann_id, ann_content)
                relations[relation.uid] = relation
            elif ann.startswith("A"):
                attribute = _parse_attribute(ann_id, ann_content)
                attributes[attribute.uid] = attribute
            elif ann.startswith("#"):
                note = _parse_note(ann_id, ann_content)
                notes[note.uid] = note
        except ValueError as err:
            logger.warning(err)
            logger.warning("Ignore annotation %s at line %s", ann_id, line_number)

    # Process groups
    groups = None
    if detect_groups:
        groups: dict[str, Grouping] = {}
        grouping_relations = {r.uid: r for r in relations.values() if r.type in GROUPING_RELATIONS}

        for entity in entities.values():
            if entity.type in GROUPING_ENTITIES:
                items = [
                    entities[relation.obj] for relation in grouping_relations.values() if relation.subj == entity.uid
                ]
                groups[entity.uid] = Grouping(entity.uid, entity.type, items)

    return BratDocument(entities, relations, attributes, notes, groups)


def _parse_entity(entity_id: str, entity_content: str) -> BratEntity:
    """Parse the brat entity string into an Entity structure.

    Parameters
    ----------
    entity_id : str
        The ID defined in the brat annotation (e.g.,`T12`)
    entity_content : str
        The string content for this ID to parse
         (e.g., `Temporal-Modifier 116 126\thistory of`)

    Returns
    -------
    BratEntity
        The dataclass object representing the entity

    Raises
    ------
    ValueError
        Raises when the entity can't be parsed
    """
    try:
        tag_and_spans, text = entity_content.strip().split("\t", maxsplit=1)
        text = text.replace("\t", " ")  # Remove tabs in text

        tag, spans_text = tag_and_spans.split(" ", maxsplit=1)
        span_list = spans_text.split(";")
        spans: list[tuple[int, int]] = []
        for span in span_list:
            start_s, end_s = span.split()
            start, end = int(start_s), int(end_s)
            spans.append((start, end))
        return BratEntity(entity_id.strip(), tag.strip(), spans, text)
    except ValueError as err:
        msg = "Impossible to parse entity."
        raise ValueError(msg) from err


def _parse_relation(relation_id: str, relation_content: str) -> BratRelation:
    """Parse the annotation string into a Relation structure.

    Parameters
    ----------
    relation_id : str
        The ID defined in the brat annotation (e.g., R12)
    relation_content : str
        The relation text content. (e.g., `Modified-By Arg1:T8 Arg2:T6\t`)

    Returns
    -------
    BratRelation
        The dataclass object representing the relation

    Raises
    ------
    ValueError
        Raises when the relation can't be parsed
    """
    try:
        relation, subj, obj = relation_content.strip().split()
        subj = subj.replace("Arg1:", "")
        obj = obj.replace("Arg2:", "")
    except ValueError as err:
        msg = "Impossible to parse relation."
        raise ValueError(msg) from err

    if subj.startswith("E") or obj.startswith("E"):
        msg = "Relations between events are not supported."
        raise ValueError(msg)

    return BratRelation(relation_id.strip(), relation.strip(), subj.strip(), obj.strip())


def _parse_attribute(attribute_id: str, attribute_content: str) -> BratAttribute:
    """Parse the annotation string into an Attribute structure.

    Parameters
    ----------
    attribute_id : str
        The attribute ID defined in the annotation. (e.g., `A1`)
    attribute_content : str
         The attribute text content. (e.g., `Tense T19 Past-Ended`)

    Returns
    -------
    BratAttribute
        The dataclass object representing the attribute

    Raises
    ------
    ValueError
        Raises when the attribute can't be parsed
    """
    attribute_arguments = attribute_content.strip().split(" ", maxsplit=2)

    try:
        attribute_name, attribute_target = attribute_arguments[:2]
    except (IndexError, ValueError) as err:
        msg = "Impossible to parse attribute."
        raise ValueError(msg) from err

    try:
        attribute_value = attribute_arguments[2].strip()
    except IndexError:
        attribute_value = None

    return BratAttribute(
        attribute_id.strip(),
        attribute_name.strip(),
        attribute_target.strip(),
        attribute_value,
    )


def _parse_note(note_id: str, note_content: str) -> BratNote:
    """Parse the annotation string into an Note structure.

    Parameters
    ----------
    note_id : str
        The note ID defined in the annotation. (e.g., `#1`)
    note_content : str
        The note text content. (e.g., `AnnotatorNotes T10	C0011849`)

    Returns
    -------
    BratNote
        The dataclass object representing the note

    Raises
    ------
    ValueError
        Raises when the note can't be parsed
    """
    parts = note_content.split("\t", maxsplit=1)
    try:
        note_arguments, note_value = parts
    except ValueError as err:
        msg = "Impossible to parse the input note"
        raise ValueError(msg) from err

    parts = note_arguments.split(" ", maxsplit=1)
    try:
        note_type, note_target = parts
    except ValueError as err:
        msg = "Impossible to parse the input note"
        raise ValueError(msg) from err

    return BratNote(
        uid=note_id.strip(),
        target=note_target.strip(),
        value=note_value.strip(),
        type=note_type.strip(),
    )
