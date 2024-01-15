__all__ = ["TextItem", "get_text_items", "Generator", "Prefixer", "Splitter", "Merger"]

from medkit.core import OperationDescription, generate_id


class TextItem:
    def __init__(self, text):
        """Mock text item with uid"""
        self.uid = generate_id()
        self.text = text


def get_text_items(nb_items):
    return [TextItem(f"This is the text item number {i}.") for i in range(nb_items)]


class Generator:
    def __init__(self, prov_tracer=None):
        """Mock operation generating text items"""
        self.uid = generate_id()
        self.prov_tracer = prov_tracer
        self.description = OperationDescription(uid=self.uid, name="Generator")

    def generate(self, nb_items):
        items = get_text_items(nb_items)

        if self.prov_tracer is not None:
            for item in items:
                self.prov_tracer.add_prov(item, self.description, source_data_items=[])
        return items


class Prefixer:
    def __init__(self, prov_tracer=None):
        """Mock operation prefixing items"""
        self.uid = generate_id()
        self.prov_tracer = prov_tracer
        self.description = OperationDescription(uid=self.uid, name="Prefixer")

    def prefix(self, items):
        prefixed_items = []
        for item in items:
            prefixed_item = TextItem("Hello! " + item.text)
            prefixed_items.append(prefixed_item)
            if self.prov_tracer is not None:
                self.prov_tracer.add_prov(prefixed_item, self.description, source_data_items=[item])
        return prefixed_items


class Splitter:
    def __init__(self, prov_tracer=None):
        """Mock operation splitting items"""
        self.uid = generate_id()
        self.prov_tracer = prov_tracer
        self.description = OperationDescription(uid=self.uid, name="Splitter")

    def split(self, items):
        split_items = []
        for item in items:
            half = len(item.text) // 2
            left_item = TextItem(item.text[:half])
            split_items.append(left_item)
            right_item = TextItem(item.text[half:])
            split_items.append(right_item)

            if self.prov_tracer is not None:
                self.prov_tracer.add_prov(left_item, self.description, source_data_items=[item])
                self.prov_tracer.add_prov(right_item, self.description, source_data_items=[item])
        return split_items


class Merger:
    """Mock operation merging items"""

    def __init__(self, prov_tracer=None):
        self.uid = generate_id()
        self.prov_tracer = prov_tracer
        self.description = OperationDescription(uid=self.uid, name="Merger")

    def merge(self, items):
        text = "".join(s.text for s in items)
        merged_item = TextItem(text)
        if self.prov_tracer is not None:
            self.prov_tracer.add_prov(merged_item, self.description, source_data_items=items)
        return merged_item
