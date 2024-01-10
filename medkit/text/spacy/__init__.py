"""This package needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[spacy]`.
"""

__all__ = [
    "SpacyDocPipeline",
    "SpacyPipeline",
    # not imported
    "spacy_utils",
    "displacy_utils",
]

# Verify that spacy is installed
from medkit.core.utils import modules_are_available

if not modules_are_available(["spacy"]):
    raise ImportError("Requires spacy install for importing medkit.text.spacy module")

from .doc_pipeline import SpacyDocPipeline
from .pipeline import SpacyPipeline

if modules_are_available(["edsnlp"]):
    __all__ += ["edsnlp"]
