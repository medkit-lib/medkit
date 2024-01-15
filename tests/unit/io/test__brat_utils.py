import pathlib

import pytest

from medkit.io._brat_utils import (
    AttributeConf,
    BratAnnConfiguration,
    BratAttribute,
    BratEntity,
    BratRelation,
    RelationConf,
    _parse_attribute,
    _parse_entity,
    _parse_note,
    _parse_relation,
    parse_file,
)


def test__parse_entity():
    brat_entity = "T1	medication 36  46	Lisinopril\t"
    entity_id, entity_content = brat_entity.split("\t", maxsplit=1)
    entity = _parse_entity(entity_id, entity_content)
    assert entity.uid == "T1"
    assert entity.type == "medication"
    assert entity.span == [(36, 46)]
    assert entity.text == "Lisinopril"


def test__parse_entity_leading_space():
    brat_entity = "T1	medication 36  46	 Lisinopril\t"
    entity_id, entity_content = brat_entity.split("\t", maxsplit=1)
    entity = _parse_entity(entity_id, entity_content)
    assert entity.text == " Lisinopril"


def test__parse__entity_discontinued_span():
    brat_entity = "T6	vitamin 251 260;263 264	vitamins D"
    entity_id, entity_content = brat_entity.split("\t", maxsplit=1)
    entity = _parse_entity(entity_id, entity_content)
    assert entity.span == [(251, 260), (263, 264)]


def test__parse_entity_error():
    brat_entity = "T1	medication 36 46 Lisinopril"
    entity_id, entity_content = brat_entity.split("\t", maxsplit=1)
    with pytest.raises(ValueError, match="Impossible to parse entity."):
        _parse_entity(entity_id, entity_content)


def test__parse_relation():
    brat_relation = "R1	treats Arg1:T1 Arg2:T3"
    relation_id, relation_content = brat_relation.split("\t", maxsplit=1)
    relation = _parse_relation(relation_id, relation_content)
    assert relation.uid == "R1"
    assert relation.type == "treats"
    assert relation.subj == "T1"
    assert relation.obj == "T3"


def test__parse_relation_error():
    brat_relation = "R1	treats  Arg1:T1\t"
    relation_id, relation_content = brat_relation.split("\t", maxsplit=1)
    with pytest.raises(ValueError, match="Impossible to parse relation."):
        _parse_relation(relation_id, relation_content)


def test__parse_relation_in_events_error():
    brat_relation = "R1	treats Arg1:E1 Arg2:T1"
    relation_id, relation_content = brat_relation.split("\t", maxsplit=1)
    with pytest.raises(ValueError, match="Relations between events are not supported."):
        _parse_relation(relation_id, relation_content)


def test__parse_attribute():
    brat_attribute = "A3	 antecedent T4"
    attribute_id, attribute_content = brat_attribute.split("\t", maxsplit=1)
    attribute = _parse_attribute(attribute_id, attribute_content)
    assert attribute.uid == "A3"
    assert attribute.type == "antecedent"
    assert attribute.target == "T4"


def test__parse_attribute_value():
    brat_attribute = "A2	severity R2  normal "
    attribute_id, attribute_content = brat_attribute.split("\t", maxsplit=1)
    attribute = _parse_attribute(attribute_id, attribute_content)
    assert attribute.value == "normal"


def test__parse_attribute_error():
    brat_attribute = "A2	severity "
    attribute_id, attribute_content = brat_attribute.split("\t", maxsplit=1)
    with pytest.raises(ValueError, match="Impossible to parse attribute."):
        _parse_attribute(attribute_id, attribute_content)


def test__parse_note():
    brat_note = "#1	AnnotatorNotes T20	C0011849"
    note_id, note_content = brat_note.split("\t", maxsplit=1)
    note = _parse_note(note_id, note_content)
    assert note.uid == "#1"
    assert note.type == "AnnotatorNotes"
    assert note.target == "T20"


def test_parse_file():
    test_file = pathlib.Path("tests/data/brat/1_example.ann")
    doc = parse_file(str(test_file))
    entity = BratEntity(
        uid="T1",
        type="medication",
        span=[(36, 46)],
        text="Lisinopril",
    )
    assert entity in doc.entities.values()
    relation = BratRelation(uid="R1", type="treats", subj="T1", obj="T3")
    assert relation in doc.relations.values()
    attribute = BratAttribute(uid="A1", type="negation", target="R1")
    assert attribute in doc.attributes.values()


def test_document_get_augmented_entities():
    test_file = pathlib.Path("tests/data/brat/2_augmented_entities.ann")
    doc = parse_file(test_file)
    augmented_entities = doc.get_augmented_entities()
    assert "T4" in augmented_entities
    entity = augmented_entities["T4"]
    assert entity.text == "entity1 entity2"
    assert entity.type == "And-Group"
    assert entity.span == [(30, 37), (120, 127)]
    relation1 = BratRelation(uid="R1", type="And", subj="T4", obj="T1")
    relation2 = BratRelation(uid="R3", type="Or", subj="T5", obj="T4")
    assert relation1 in entity.relations_from_me
    assert relation2 in entity.relations_to_me
    attribute = BratAttribute(uid="A1", type="attribute", target="T4")
    assert attribute in entity.attributes


def test_document_grouping():
    test_file = pathlib.Path("tests/data/brat/2_augmented_entities.ann")
    doc = parse_file(test_file, detect_groups=True)
    assert "T1" not in doc.groups
    # Test And-Group
    assert "T4" in doc.groups
    and_group = doc.groups["T4"]
    assert and_group.type == "And-Group"
    entity1 = BratEntity(uid="T1", type="label1", span=[(30, 37)], text="entity1")
    assert entity1 in and_group.items
    # Test Or-Group
    assert "T5" in doc.groups
    or_group = doc.groups["T5"]
    assert or_group.type == "Or-Group"
    entity3 = BratEntity(uid="T3", type="label3", span=[(140, 147)], text="entity3")
    assert entity3 in or_group.items


def test_attribute_conf_file():
    conf_file = BratAnnConfiguration()
    # generate a configuration line for brat attributes
    # a brat entity has an attribute 'severity' value 'normal')
    attr_conf = AttributeConf(from_entity=True, type="severity", value="normal")
    conf_file.add_attribute_type(attr_conf)
    # another brat entity has an attribute 'severity' value 'low')
    # we add a new attribute configuration in the config file
    attr_conf = AttributeConf(from_entity=True, type="severity", value="low")
    conf_file.add_attribute_type(attr_conf)

    # finally a brat relation has an attribute 'severity' value 'inter'
    attr_conf = AttributeConf(from_entity=False, type="severity", value="inter")
    conf_file.add_attribute_type(attr_conf)

    entity_attrs = conf_file.attr_entity_values
    assert list(entity_attrs.keys()) == ["severity"]
    assert entity_attrs["severity"] == ["low", "normal"]
    assert (
        conf_file._attribute_to_str("severity", entity_attrs["severity"], True)
        == "severity\tArg:<ENTITY>, Value:low|normal"
    )

    relation_attrs = conf_file.attr_relation_values
    assert list(relation_attrs.keys()) == ["severity"]
    assert relation_attrs["severity"] == ["inter"]
    assert (
        conf_file._attribute_to_str("severity", relation_attrs["severity"], False)
        == "severity\tArg:<RELATION>, Value:inter"
    )


def test_relation_conf_file():
    conf_file = BratAnnConfiguration()
    # generate a configuration line for brat relations
    # a brat relation has subj type 'medicament_1' and obj type 'disease'
    relation_conf = RelationConf(type="treats", arg1="medicament_1", arg2="disease")
    conf_file.add_relation_type(relation_conf)
    # the same relation exists between subj type 'medicament_2' and obj type 'disease'
    # we update the relation configuration
    relation_conf = RelationConf(type="treats", arg1="medicament_2", arg2="disease")
    conf_file.add_relation_type(relation_conf)

    relation_conf_args1 = conf_file.rel_types_arg_1
    relation_conf_args2 = conf_file.rel_types_arg_2

    assert list(relation_conf_args1.keys()) == ["treats"]
    assert list(relation_conf_args2.keys()) == ["treats"]
    assert relation_conf_args1["treats"] == ["medicament_1", "medicament_2"]
    assert relation_conf_args2["treats"] == ["disease"]
    assert (
        conf_file._relation_to_str("treats", relation_conf_args1["treats"], relation_conf_args2["treats"])
        == "treats\tArg1:medicament_1|medicament_2, Arg2:disease"
    )


TEST_CONFIG = [
    (0, [], "severity\tArg:<ENTITY>"),
    (2, ["low", "normal"], "severity\tArg:<ENTITY>, Value:low|normal"),
    (
        10,
        ["low", "normal", "other", "other_V"],
        "severity\tArg:<ENTITY>, Value:low|normal|other|other_V",
    ),
]


@pytest.mark.parametrize(
    ("top_values_by_attr", "expected_values", "expected_str"),
    TEST_CONFIG,
    ids=["no_values", "max_2_values", "all_values"],
)
def test_attribute_entity_conf_file_top_values(top_values_by_attr, expected_values, expected_str):
    # testing limit of values in attr config
    # an attribute may have many values,'top_values_by_attr' allow to
    # limit that number. Only the 'n' most common values will be shown in the config (max)
    conf_file = BratAnnConfiguration(top_values_by_attr=top_values_by_attr)

    # add values into the config
    # the value 'normal' appears 10 times, 'low' 5 times, and 'other' and 'other_V' once
    for _ in range(10):
        attr_conf = AttributeConf(from_entity=True, type="severity", value="normal")
        conf_file.add_attribute_type(attr_conf)

    for _ in range(5):
        attr_conf = AttributeConf(from_entity=True, type="severity", value="low")
        conf_file.add_attribute_type(attr_conf)

    attr_conf = AttributeConf(from_entity=True, type="severity", value="other")
    conf_file.add_attribute_type(attr_conf)

    attr_conf = AttributeConf(from_entity=True, type="severity", value="other_V")
    conf_file.add_attribute_type(attr_conf)

    # check values
    entity_attrs = conf_file.attr_entity_values
    assert list(entity_attrs.keys()) == ["severity"]
    assert entity_attrs["severity"] == expected_values
    assert conf_file._attribute_to_str("severity", entity_attrs["severity"], True) == expected_str
