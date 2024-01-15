from medkit.core import AnnotationContainer, generate_id


class _MockAnnotation:
    def __init__(self, label, value, keys=None):
        if keys is None:
            keys = set()

        self.uid = generate_id()
        self.label = label
        self.value = value
        self.keys = keys


def test_basic():
    "Basic usage, add annotations and retrieve them"
    anns = AnnotationContainer(doc_id="id")

    # empty container
    assert anns.get() == []
    assert anns.get(label="cui") == []
    assert list(iter(anns)) == []  # __iter__()
    assert len(anns) == 0  # __len__()

    ann_1 = _MockAnnotation("name", "Bob")
    anns.add(ann_1)
    ann_2 = _MockAnnotation("topic", "Cancer")
    anns.add(ann_2)
    ann_3 = _MockAnnotation("topic", "Chemotherapy", keys={"entities"})
    anns.add(ann_3)

    assert anns.get() == [ann_1, ann_2, ann_3]
    # label filtering
    assert anns.get(label="name") == [ann_1]
    assert anns.get(label="topic") == [ann_2, ann_3]
    assert anns.get(label="misc") == []
    # key filtering
    assert anns.get(key="entities") == [ann_3]
    assert anns.get(key="sentences") == []
    # combined label and key filtering
    assert anns.get(label="topic", key="entities") == [ann_3]
    assert anns.get(label="name", key="entities") == []

    assert len(anns) == 3  # _len__()
    assert list(iter(anns)) == anns.get()  # __iter__()
    assert anns[0] is ann_1  # __getitem__()
    assert anns[1:3] == [ann_2, ann_3]  # __getitem__()

    assert anns.get_by_id(ann_1.uid) == ann_1
