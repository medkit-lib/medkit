import abc
import dataclasses

import pytest

from medkit.core import dict_conv


@dataclasses.dataclass
class _MockAttribute(dict_conv.SubclassMapping):
    """Base non-abstract with its own from_dict()/to_dict() methods, but also
    handling dispatch to the from_dict() method of a subclass
    """

    label: str

    def __init_subclass__(cls):
        _MockAttribute.register_subclass(cls)
        super().__init_subclass__()

    def to_dict(self):
        attr_dict = {"label": self.label}
        dict_conv.add_class_name_to_data_dict(self, attr_dict)
        return attr_dict

    @classmethod
    def from_dict(cls, attr_dict):
        subclass = cls.get_subclass_for_data_dict(attr_dict)
        if subclass is not None:
            return subclass.from_dict(attr_dict)

        return cls(label=attr_dict["label"])


@dataclasses.dataclass
class _MockSubAttribute(_MockAttribute):
    """Subclass inheriting parent from_dict()/to_dict() methods"""


@dataclasses.dataclass
class _MockSubAttributeWithValue(_MockAttribute):
    """Subclass with its own from_dict()/to_dict() methods"""

    value: bool

    def to_dict(self):
        attr_dict = {"label": self.label, "value": self.value}
        dict_conv.add_class_name_to_data_dict(self, attr_dict)
        return attr_dict

    @classmethod
    def from_dict(cls, attr_dict):
        return cls(label=attr_dict["label"], value=attr_dict["value"])


@dataclasses.dataclass
class _MockAnnotation(abc.ABC, dict_conv.SubclassMapping):
    """Base abstract class without its own from_dict()/to_dict() methods, but still
    handling dispatch to the from_dict() method of a subclass
    """

    label: str

    def __init_subclass__(cls):
        _MockAnnotation.register_subclass(cls)
        super().__init_subclass__()

    @classmethod
    def from_dict(cls, attr_dict):
        subclass = cls.get_subclass_for_data_dict(attr_dict)
        if subclass is not None:
            return subclass.from_dict(attr_dict)

        raise NotImplementedError


@dataclasses.dataclass
class _MockSubAnnotationWithText(_MockAnnotation):
    """Subclass with its own from_dict()/to_dict() methods"""

    text: str

    def to_dict(self):
        ann_dict = {"label": self.label, "text": self.text}
        dict_conv.add_class_name_to_data_dict(self, ann_dict)
        return ann_dict

    @classmethod
    def from_dict(cls, ann_dict):
        return cls(label=ann_dict["label"], text=ann_dict["text"])


_DOCUMENT_DICT = {dict_conv._CLASS_NAME_KEY: "Document", "text": "Hello"}


def test_parent_concrete_class():
    # parent class
    attr = _MockAttribute(label="is_negated")
    attr_dict = attr.to_dict()
    attr_alt = _MockAttribute.from_dict(attr_dict)
    assert attr_alt == attr
    # using a dict with unknown class name should raise
    with pytest.raises(ValueError, match="Received a data dict with class_name *"):
        _MockAttribute.from_dict(_DOCUMENT_DICT)

    # child class with inherited from_dict()/to_dict()
    attr = _MockSubAttribute(label="is_negated")
    attr_dict = attr.to_dict()
    attr_alt = _MockSubAttribute.from_dict(attr_dict)
    assert attr_alt == attr
    # calling from_dict() on base class should also work
    attr_alt = _MockAttribute.from_dict(attr_dict)
    assert attr_alt == attr
    # NB: using a dict with unknown class name is not checked for

    # child class with own from_dict()/to_dict()
    attr = _MockSubAttributeWithValue(label="is_negated", value=True)
    attr_dict = attr.to_dict()
    attr_alt = _MockSubAttributeWithValue.from_dict(attr_dict)
    assert attr_alt == attr
    # calling from_dict() on base class should also work
    attr_alt = _MockAttribute.from_dict(attr_dict)
    assert attr_alt == attr
    # NB: using a dict with unknown class name is not checked for


def test_parent_abstract_class():
    # parent class
    # using a dict with unknown class name should raise
    with pytest.raises(ValueError, match="Received a data dict with class_name *"):
        _MockAnnotation.from_dict(_DOCUMENT_DICT)
    # trying to instantiate abstract class should fail
    ann_dict = {dict_conv._CLASS_NAME_KEY: dict_conv.get_class_name(_MockAnnotation)}
    with pytest.raises(NotImplementedError):
        ann = _MockAnnotation.from_dict(ann_dict)

    # child class with own from_dict()/to_dict()
    ann = _MockSubAnnotationWithText(label="sentence", text="Hello")
    ann_dict = ann.to_dict()
    ann_alt = _MockSubAnnotationWithText.from_dict(ann_dict)
    assert ann_alt == ann
    # calling from_dict() on base class should also work
    ann_alt = _MockAnnotation.from_dict(ann_dict)
    assert ann_alt == ann
    # NB: using a dict with unknown class name is not checked for
