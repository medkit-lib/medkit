import pytest

from medkit.core import Attribute, AttributeContainer


def test_basic():
    attrs = AttributeContainer(owner_id="id")

    # empty container
    assert len(attrs.get()) == 0
    assert len(attrs.get(label="cui")) == 0
    assert len(attrs) == 0  # __len__()
    assert list(iter(attrs)) == []  # __iter__()

    attr_1 = Attribute(label="negation", value=False)
    attrs.add(attr_1)
    attr_2 = Attribute(label="cui", value="C0011854")
    attrs.add(attr_2)
    attr_3 = Attribute(label="cui", value="C0011855")
    attrs.add(attr_3)

    assert attrs.get() == [attr_1, attr_2, attr_3]
    assert attrs.get(label="negation") == [attr_1]
    assert attrs.get(label="cui") == [attr_2, attr_3]
    assert len(attrs) == 3  # _len__()
    assert list(iter(attrs)) == attrs.get()  # __iter__()
    assert attrs[0] is attr_1  # __getitem__()
    assert attrs[1:3] == [attr_2, attr_3]  # __getitem__()


def test_add_same_attr():
    attrs = AttributeContainer(owner_id="id")
    attr = Attribute(label="negation", value=False)
    attrs.add(attr)

    with pytest.raises(ValueError, match="Attribute with uid .* already attached to annotation"):
        attrs.add(attr)
