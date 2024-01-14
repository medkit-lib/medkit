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
    msg = "Requires spacy install for importing medkit.text.spacy module"
    raise ImportError(msg)

from medkit.text.spacy.doc_pipeline import SpacyDocPipeline
from medkit.text.spacy.pipeline import SpacyPipeline

if modules_are_available(["edsnlp"]):
    __all__ += ["edsnlp"]
