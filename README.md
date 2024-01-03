# medkit

![medkit logo](https://github.com/medkit-lib/medkit/blob/main/docs/img/medkit_logo.png?raw=true)

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CI      | [![docs status](https://readthedocs.org/projects/medkit/badge/?version=latest)](https://medkit.readthedocs.io/en/latest/) [![lint status](https://github.com/medkit-lib/medkit/actions/workflows/lint.yaml/badge.svg)](https://github.com/medkit-lib/medkit/actions/workflows/lint.yaml) [![test: status](https://github.com/medkit-lib/medkit/actions/workflows/test.yaml/badge.svg)](https://github.com/medkit-lib/medkit/actions/workflows/test.yaml)              |
| Package | [![PyPI version](https://img.shields.io/pypi/v/medkit-lib.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/medkit-lib/) [![PyPI downloads](https://img.shields.io/pypi/dm/medkit-lib.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/medkit-lib/) [![PyPI Python versions](https://img.shields.io/pypi/pyversions/medkit-lib.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/medkit-lib/) |
| Project | [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://spdx.org/licenses/MIT.html) [![Formatter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Project: Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://hatch.pypa.io)                                                                              |

----

`medkit` is a toolkit for a learning health system, developed by the [HeKA research team](https://team.inria.fr/heka).

This python library aims at:

1. Facilitating the manipulation of healthcare data of various modalities (e.g., structured, text, audio data)
for the extraction of relevant features.

2. Developing supervised models from these various modalities for decision support in healthcare.

## Installation

To install `medkit` with basic functionalities:

```console
pip install medkit-lib
```

To install `medkit` with all its optional features:

```console
pip install medkit-lib[all]
```

## Example

A basic named-entity recognition pipeline using `medkit`:

```python
# 1. Define individual operations.
from medkit.text.preprocessing import CharReplacer, LIGATURE_RULES, SIGN_RULES
from medkit.text.segmentation import SentenceTokenizer, SyntagmaTokenizer
from medkit.text.context.negation_detector import NegationDetector
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher

# Preprocessing
char_replacer = CharReplacer(rules=LIGATURE_RULES + SIGN_RULES)
# Segmentation
sent_tokenizer = SentenceTokenizer(output_label="sentence")
synt_tokenizer = SyntagmaTokenizer(output_label="syntagma")
# Negation detection
neg_detector = NegationDetector(output_label="is_negated")
# Entity recognition
entity_matcher = HFEntityMatcher(model="my-BERT-model", attrs_to_copy=["is_negated"])

# 2. Combine operations into a pipeline.
from medkit.core.pipeline import Pipeline, PipelineStep

ner_pipeline = Pipeline(
    input_keys=["full_text"],
    output_keys=["entities"],
    steps=[
        PipelineStep(char_replacer, input_keys=["full_text"], output_keys=["clean_text"]),
        PipelineStep(sent_tokenizer, input_keys=["clean_text"], output_keys=["sentences"]),
        PipelineStep(synt_tokenizer, input_keys=["sentences"], output_keys=["syntagmas"]),
        PipelineStep(neg_detector, input_keys=["syntagmas"], output_keys=[]),
        PipelineStep(entity_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
    ],
)

# 3. Run the NER pipeline on a BRAT document.
from medkit.io import BratInputConverter

docs = BratInputConverter().load(path="/path/to/dataset/")
entities = ner_pipeline.run([doc.raw_segment for doc in docs])
```

## Getting started

To get started with `medkit`, please checkout our [documentation](https://medkit.readthedocs.io/).

This documentation also contains tutorials and examples showcasing the use of `medkit` for different tasks.

## Contributing

Thank you for your interest into medkit !

We'll be happy to get your inputs !

If your problem has not been reported by another user, please open an
[issue](https://github.com/medkit-lib/medkit/issues), whether it's for:

* reporting a bug, 
* discussing the current state of the code, 
* submitting a fix, 
* proposing new features, 
* or contributing to documentation, ...

If you want to propose a pull request, you can read [CONTRIBUTING.md](./CONTRIBUTING.md).

## Contact

Feel free to contact us by sending an email to [medkit-maintainers@inria.fr](mailto:medkit-maintainers@inria.fr).
