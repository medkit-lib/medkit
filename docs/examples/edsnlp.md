---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Using EDS-NLP with medkit 

[EDS-NLP](https://aphp.github.io/edsnlp/) provides a set of
[spaCy](https://spacy.io/) components that are used to extract information from
clinical notes written in French. Because medkit is spaCy-compatible, using
EDS-NLP within medkit is supported, as we will see.

To follow this tutorial, you will need to install medkit spaCy support and
EDS-NLP with
```{code-cell} ipython3
:tags: [remove-output]

pip install 'medkit-lib[edsnlp]'
```
## Running an EDS-NLP spaCy pipeline on entire documents

We will need a sample text document to annotate:

```{code-cell} ipython3
from medkit.core.text import TextDocument

text = """COMPTE RENDU D'HOSPITALISATION
Monsieur Jean Dupont a été hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme

ANTÉCÉDENTS
Peut-être atteint de Covid19 en aout 2020"""
doc = TextDocument(text)
```

and a spaCy pipeline with a few EDS-NLP components:

```{code-cell} ipython3
:tags: [remove-output]

import spacy

nlp = spacy.blank("eds")

# General-purpose components
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sentences")
# Entity extraction
nlp.add_pipe("eds.covid")
nlp.add_pipe("eds.dates")
# Context detection
nlp.add_pipe("eds.negation")
nlp.add_pipe("eds.hypothesis")
```

The `eds.normalizer` and `eds.sentences` components do some pre-processing,
`eds.covid` and `eds.dates` perform entity matching and create some spaCy
entities and spans, and `eds.negation` and `eds.hypothesis` attach some context
attributes to these entities and spans.

To be used within medkit, the pipeline could be wrapped into a generic
{class}`~medkit.text.spacy.SpacyDocPipeline` operation. But medkit also provides
a dedicated {class}`~.EDSNLPDocPipeline` operation, with some additional support
for specific EDS-NLP components:


```{code-cell} ipython3
:tags: [remove-output]
from medkit.text.spacy.edsnlp import EDSNLPDocPipeline

eds_nlp_pipeline = EDSNLPDocPipeline(nlp)
```

The operation is executed by applying its `run()` method on a list of documents:

```{code-cell} ipython3
eds_nlp_pipeline.run([doc])
```

Let's look at the entities and segments that were found:

```{code-cell} ipython3
for entity in doc.anns.entities:
    print(f"{entity.label}: {entity.text!r}")
for segment in doc.anns.segments:
    print(f"{segment.label}: {segment.text!r}")
```

Here are the attributes attached to the `"covid"` entity:

```{code-cell} ipython3
entity = doc.anns.get_entities(label="covid")[0]
for attr in entity.attrs:
    print(f"{attr.label}={attr.value}")
```

and the attributes of the first `"dates"` segment:

```{code-cell} ipython3
date_seg = doc.anns.get_segments(label="dates")[0]
for attr in date_seg.attrs:
    print(f"{attr.label}={attr.value}")
```

Let's now examine more closely the `"date"` attribute:

```{code-cell} ipython3
date_seg = doc.anns.get_segments(label="dates")[0]
date_attr = date_seg.attrs.get(label="date")[0]
date_attr
```

This attribute is an instance of {class}`~medkit.text.ner.DateAttribute`, a
subclass of {class}`~medkit.core.Attribute`.It has `year`, `month`, `day` (etc)
fields containing the different parts of the date that was detected, as well as
a normalized string representation in its `value` field:
```{code-cell} ipython3
date_attr.value
```

One of the benefits of using {class}`~.EDSNLPDocPipeline` instead of
{class}`~medkit.text.spacy.SpacyDocPipeline` is that some special EDS-NLP
attributes are automatically converted to a corresponding
{class}`~medkit.core.Attribute` subclass.

Here are the supported EDS-NLP attributes values and the corresponding medkit classes:
- `AdicapCode` (created by `eds.adicap`): {class}`medkit.text.ner.ADICAPNormAttribute`
- `TNM` (created by `eds.tnm`): {class}`medkit.text.ner.tnm_attribute.TNMAttribute`
- `AbsoluteDate` (created by `eds.dates`): {class}`medkit.text.ner.DateAttribute`
- `RelativeDate` (created by `eds.dates`): {class}`medkit.text.ner.RelativeDateAttribute`
- `Duration` (created by `eds.dates`): {class}`medkit.text.ner.DurationAttribute`

```{note}
The transformations performed by {class}`~.EDSNLPDocPipeline` can be overridden
or extended with the `medkit_attribute_factories` init parameter. For a list of
all the default transformations, see
{const}`~medkit.text.spacy.edsnlp.DEFAULT_ATTRIBUTE_FACTORIES` and corresponding
functions in {mod}`medkit.text.spacy.edsnlp`.
```

## Running an EDL-NLP spaCy pipeline at the annotation level

So far, we have wrapped a spaCy pipeline and executed it on an entire document
with {class}`~.EDSNLPDocPipeline`. But it is also possible to run the spaCy
pipeline on text annotations instead of a document with
{class}`~.EDSNLPPipeline`. To illustrate this, let's create a medkit pipeline
using pure medkit operations for sentence tokenization and entity matching, and
EDS-NLP spaCy components for covid entity matching:

```{code-cell} ipython3
:tags: [remove-output]
from medkit.core import Pipeline, PipelineStep
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule
from medkit.text.segmentation import SentenceTokenizer
from medkit.text.spacy.edsnlp import EDSNLPPipeline

sentence_tokenizer = SentenceTokenizer()
matcher = RegexpMatcher(rules=[RegexpMatcherRule(regexp=r"\basthme\b", label="asthme")])

nlp = spacy.blank("eds")
nlp.add_pipe("eds.covid")
eds_nlp_pipeline = EDSNLPPipeline(nlp)

pipeline = Pipeline(
    steps=[
        PipelineStep(operation=sentence_tokenizer, input_keys=["full_text"], output_keys=["sentences"]),
        PipelineStep(operation=matcher, input_keys=["sentences"], output_keys=["entities"]),
        PipelineStep(operation=eds_nlp_pipeline, input_keys=["sentences"], output_keys=["entities"]),
    ],
    input_keys=["full_text"],
    output_keys=["entities"],
)
```

```{code-cell} ipython3
doc = TextDocument(text)
entities = pipeline.run([doc.raw_segment])
for entity in entities:
    print(f"{entity.label}: {entity.text!r}")
```

For more information about advanced usage of {class}`~.EDSNLPDocPipeline` and
{class}`~.EDSNLPPipeline`, you may refer to the API doc of
{mod}`medkit.text.spacy.edsnlp`.
