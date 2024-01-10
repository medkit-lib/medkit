__all__ = []

from medkit.core.utils import modules_are_available

# Syntactic Relation Extractor optional module
if modules_are_available(["spacy"]):
    __all__ += ["syntactic_relation_extractor"]
