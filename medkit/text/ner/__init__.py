__all__ = [
    "ADICAPNormAttribute",
    "DucklingMatcher",
    "RegexpMatcher",
    "RegexpMatcherRule",
    "RegexpMatcherNormalization",
    "RegexpMetadata",
    "SimstringMatcher",
    "SimstringMatcherRule",
    "SimstringMatcherNormalization",
    "UMLSMatcher",
    "IAMSystemMatcher",
    "MedkitKeyword",
    "DateAttribute",
    "DurationAttribute",
    "RelativeDateAttribute",
    "RelativeDateDirection",
]

from medkit.core.utils import modules_are_available

from .adicap_norm_attribute import ADICAPNormAttribute
from .date_attribute import (
    DateAttribute,
    DurationAttribute,
    RelativeDateAttribute,
    RelativeDateDirection,
)
from .duckling_matcher import DucklingMatcher
from .iamsystem_matcher import IAMSystemMatcher, MedkitKeyword
from .regexp_matcher import (
    RegexpMatcher,
    RegexpMatcherNormalization,
    RegexpMatcherRule,
    RegexpMetadata,
)
from .simstring_matcher import (
    SimstringMatcher,
    SimstringMatcherNormalization,
    SimstringMatcherRule,
)
from .umls_matcher import UMLSMatcher

# quick_umls module
if modules_are_available(["packaging", "quickumls"]):
    __all__ += ["quick_umls_matcher"]

# HF entity matcher
if modules_are_available(["torch", "transformers"]):
    __all__ += ["hf_entity_matcher", "hf_entity_matcher_trainable"]

if modules_are_available(["pandas", "torch", "transformers"]):
    __all__ += ["umls_coder_normalizer"]

if modules_are_available(["edsnlp"]):
    __all__ += ["tnm_attribute"]

if modules_are_available(["nlstruct", "torch", "huggingface_hub"]):
    __all__ += ["nlstruct_entity_matcher"]
