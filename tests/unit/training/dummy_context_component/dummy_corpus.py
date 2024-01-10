from medkit.core import Attribute
from medkit.core.text import Segment, Span

texts_training = [
    "La prise en charge thérapeutique ne sera pas modifiée par l'étude",
    "La prise en charge thérapeutique sera modifiée par l'étude",
    "Le patient est malade",
    "Le patient n'est pas malade",
    "L'objectif est de comparer l'efficacité et la tolérance de la gemcitabine",
    "Compte de rendu medical",
]
labels_training = ["neg", "pos", "pos", "neg", "pos", "pos"]

texts_eval = ["La prise en charge ne sera pas autorisée", "Le patient a une maladie"]
labels_eval = ["neg", "pos"]


def get_segment(text, value):
    segment = Segment(text=text, label="raw_segment", spans=[Span(0, len(text))])
    segment.attrs.add(Attribute(label="category", value=value))
    return segment


DUMMY_DATASETS = {
    "train": [get_segment(text, value) for text, value in zip(texts_training, labels_training)],
    "eval": [get_segment(text, value) for text, value in zip(texts_eval, labels_eval)],
}
