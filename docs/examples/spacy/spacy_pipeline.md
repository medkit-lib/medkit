---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Annotating with a Spacy pipeline
+++

This example shows how to combine **medkit** and a **spacy pipeline** to annotate medkit documents.

SpaCy has some projects in its universe with custom versions of spaCy pipeline objects.

This example uses English documents, as the pipelines we will use do not work with French documents. The aim of this example is to show how to annotate with spacy, but you could use your own custom pipelines that work with French documents.

```{code-cell} ipython3

# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/text/1-EN-version.txt
from pathlib import Path
from medkit.core.text import TextDocument

medkit_doc = TextDocument.from_file(Path("../../data/text/1-EN-version.txt"))
print(medkit_doc.text)
```

The document has a few sections describing the status of a female patient. We can start by detecting some entities. In the spacy universe, we found a connector [spacy-stanza](https://github.com/explosion/spacy-stanza) to the Stanza library.  **Stanza**[^footnote1] is a library developed by the Stanford NLP research group and has some biomedical and clinical NER models for english documents.

[^footnote1]:Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020.

```{code-cell} ipython3
:tags: [remove-output]
# install spacy-stanza
!python -m pip install spacy-stanza
```
## Annotating segments with spacy

Let's see how to create medkit entities with a nlp spacy object

### Prepare the spacy-stanza nlp pipeline

The list of available [biomedical NER packages](https://stanfordnlp.github.io/stanza/available_biomed_models.html#biomedical--clinical-ner-models).

Let's download the `i2b2` stanza package, a pretrained model to detect 'PROBLEM', 'TEST', 'TREATMENT' entities.


```{code-cell} ipython3
:tags: [remove-output]

# import spacy related modules
import stanza
import spacy_stanza

# stanza creates a nlp object in disk
# download and initialize the i2b2 pipeline
stanza.download('en', package='i2b2')
```

```{code-cell} ipython3
:tags: [remove-output]
# Define the nlp object
nlp_spacy = spacy_stanza.load_pipeline('en', package='mimic', processors={'ner': 'i2b2'})
```

### Define a medkit operation to add the entities

Medkit has the {class}`~.text.spacy.SpacyPipeline` operation, an operation that can wrap a nlp spacy object to annotate segments.

A nlp object may create many spacy annotations, you can select the spacy entities, spans and attributes that will be converted to medkit annotations. By default, all are converted into medkit annotations.

```{code-cell} ipython3
from medkit.text.spacy import SpacyPipeline

# Defines the medkit operation
medkit_stanza_matcher = SpacyPipeline(nlp=nlp_spacy)

# Detect entities using the raw segment
entities = medkit_stanza_matcher.run([medkit_doc.raw_segment])

# Add entities to the medkit document
for ent in entities:
    medkit_doc.anns.add(ent)
```

```{code-cell} ipython3
print(medkit_doc.anns.get_entities()[0])
```

That's all! We have detected entities using the biomedical model developed by the Stanford group.

Let's visualize all the detected entities.

```{code-cell} ipython3
from spacy import displacy
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy
```


```{code-cell} ipython3
# Add some colors
options_displacy = dict(colors={'TREATMENT': "#85C1E9", "PROBLEM": "#cfe2f3"})

# Format the medkit doc to displacy
displacy_data = medkit_doc_to_displacy(medkit_doc)
displacy.render(displacy_data,style="ent",manual=True, options=options_displacy)
```


## Annotating documents with spacy

Here, we already have an annotated document. We will see how to use spacy to enrich existing annotations.

Exploring the spacy universe, we found [**negspaCy**](https://spacy.io/universe/project/negspacy), a pipeline that detects negation in spacy entities. Using the 'SpacyDoc' class, we can annotate the entities of the document and add those attributes directly.

### Prepare the negspacy nlp object:

```{code-cell} ipython3
:tags: [remove-output]
# install negspacy 
!python -m pip install negspacy
```

```{code-cell} ipython3
:tags: [remove-output]
# download english model from spacy
import spacy
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
```

```{code-cell} ipython3
# Import spacy nlp object from negspacy
from negspacy.negation import Negex

# Load the EN spacy model
nlp_spacy_negex = spacy.load("en_core_web_sm",disable=["ner"]) # Disable NER by default, it can add generic entities

# Config to detect negation in the i2b2 entities
i2b2_labels = ["PROBLEM","TEST","TREATMENT"]
nlp_spacy_negex.add_pipe("negex", config={"ent_types":i2b2_labels})
```


### Define a medkit operation to add the attributes

Medkit has the {class}`~.text.spacy.SpacyDocPipeline` operation, an operation that can wrap a nlp spacy object to annotate documents.

The point is to add attributes to the entities, so we select the entities of interest and do not transfer their current attributes, as they are not needed to detect the negation.

```{code-cell} ipython3
from medkit.text.spacy import SpacyDocPipeline

# Define the spacy wrapper
negation_detector = SpacyDocPipeline(
    nlp=nlp_spacy_negex,
    medkit_labels_anns=i2b2_labels,  # entities to annotate
    medkit_attrs=[],                 # the current entity attrs are no important
)
# Run the detector
# The docPipeline automatically adds annotations to the document
# it is not necessary to add annotations as in the case of `medkit_stanza_matcher`
negation_detector.run([medkit_doc])
```

Let's see if the negation has been detected in the entities.


```{code-cell} ipython3
print(medkit_doc.anns.get_entities()[0])
```

As we can see, the entity now has an attribute called **negex** with `value=false`. Which means that the entity is not part of a negation.

Let's find the negated entities:

```{code-cell} ipython3
print("The following entities are negated: \n\n")
for entity in medkit_doc.anns.get_entities():
    # Get the negex attr
    attrs = entity.attrs.get(label="negex")

    # If the attr exists and is positive, show a message.
    if len(attrs) > 0 and attrs[0].value:
        print(entity.label,entity.text,entity.spans)
```

We can show the attribute value using displacy with more information in the labels


```{code-cell} ipython3
# enrich entity labels with [NEG] suffix
def format_entity(entity):
    label = entity.label
    negation_attr = entity.attrs.get(label="negex")[0]
    if negation_attr.value:
        return label + " [NEG]"
    return label

options_displacy = dict(colors={'TREATMENT [NEG]': "#D28E98", "PROBLEM [NEG]": "#D28E98"})

# Format the medkit doc to displacy with a entity formatter
displacy_data = medkit_doc_to_displacy(medkit_doc,entity_formatter=format_entity)
displacy.render(displacy_data,style="ent",manual=True, options=options_displacy)
```

For more information about advanced usage of spacy related operations, you may refer to the API doc of {mod}`medkit.text.spacy`.

