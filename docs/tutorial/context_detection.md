---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
metadata:
  execution:
    timeout: 300
---

# Context Detection

In this tutorial, we will use rule-based operations to attach additional contextual information to entities,
such as:
- the section in which the entity is located;
- is the entity negated;
- whether it appears as part of an hypothesis;
- whether it is related to the patient or part of their family's medical history.

Let's start by loading a medical report to work on:

```{code-cell} ipython3
from pathlib import Path
from medkit.core.text import TextDocument

# In case this notebook is executed outside medkit, download the example data with:
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/mtsamplesfr/1.txt
# and adjust the path below.
doc_file = Path("../data/mtsamplesfr/1.txt")
doc = TextDocument.from_file(doc_file)
print(doc.text)
```

## Section detection

`medkit` provides a {class}`~medkit.text.segmentation.SectionTokenizer` operation
that takes input segments containing full document texts and splits them into sections,
returning a segment for each section.

The section tokenizer is configured with a list of trigger terms
signaling the beginning of a section and corresponding section names.
`medkit` provides a [default list of sections],
but it is missing some sections featured in our document,
so we will manually define our own section rules:

[default list of sections]: https://github.com/medkit-lib/medkit/blob/main/medkit/text/segmentation/default_section_definition.yml

```{code-cell} ipython3
from medkit.text.segmentation import SectionTokenizer

# Give a definition of the sections we may encounter
# with the section name and corresponding triggers
sections_definition = {
    "current_drugs": ["MÉDICAMENTS ACTUELS"],
    "clinical_exam": ["EXAMEN DES SYSTÈMES", "EXAMEN PHYSIQUE"],
    "allergies": ["ALLERGIES"],
    "antecedents": ["ANTÉCÉDENTS DE LA MALADIE ACTUELLE", "ANTÉCÉDENTS MÉDICAUX"],
    "family_history": ["ANTÉCÉDENTS FAMILIAUX"],
    "life_style": ["MODE DE VIE"]
}
section_tokenizer = SectionTokenizer(sections_definition, output_label="section")

# The section tokenizer takes a list of segments as input
# and returns a list of segments for each section, with
# a "section" attribute containing the section name
section_segs = section_tokenizer.run([doc.raw_segment])
for section_seg in section_segs:
    section_attr = section_seg.attrs.get(label="section")[0]
    print("section", section_attr.value)
    print(section_seg.text, end="\n\n\n")
```

## Sentence splitting

We have covered sentence splitting [previously](../user_guide/first_steps.md),
and will reuse the same code, with a little addition:
we want the section information to be propagated onto the sentences,
i.e. we want to be able to tell in which section a sentence belongs.

For this, we will use the `attrs_to_copy` init parameter.
It takes a list of labels that we want to copy from the input segments
to the new sentences segments created by the operation.
Here, we will use it to copy the "section" attribute of the section segments
(which has the section name as value):

```{code-cell} ipython3
from medkit.text.segmentation import SentenceTokenizer

sentence_tokenizer = SentenceTokenizer(
    output_label="sentence",
    keep_punct=True,
    split_on_newlines=True,
    # Copy the "section" attribute
    attrs_to_copy=["section"],
)

# Run the sentence tokenizer on the section segments,
# not on the full text
sentence_segs = sentence_tokenizer.run(section_segs)

for sentence_seg in sentence_segs:
    # Retrieve the copied section attribute
    section_attr = sentence_seg.attrs.get(label="section")[0]
    print("section:", section_attr.value)
    print(sentence_seg.text, end="\n\n")
```

## Family history detection

In this document, we have a section dedicated to family medical history,
but this is not always the case.
To handle this, `medkit` provides a {class}`~medkit.text.context.FamilyDetector` operation
based on regular expressions.
It is somewhat similar to {class}`~medkit.text.ner.RegexpMatcher`
encountered [previously](./entity_matching.md#regular-expression-matching),
but instead of returning entities, it attaches attributes to the segments it receives,
with a boolean value indicating whether it mentions family history.

Like most rule-based operations, `FamilyDetector` comes with [predefined rules]
that will be used by default if none is provided.
For the sake of learning, we will manually create a few rules:

[predefined rules]: https://github.com/medkit-lib/medkit/blob/main/medkit/text/context/family_detector_default_rules.yml

```{code-cell} ipython3
from medkit.text.context import FamilyDetector, FamilyDetectorRule

family_rule_1 = FamilyDetectorRule(
    # Pattern to search inside each input segment.
    # If the pattern is found, the segment will be flagged
    # as being related to family history
    regexp=r"\bfamille\b",
    # Optional exclusions patterns: if found,
    # the segment won't be flagged
    # (Exclusion regexps are also supported for RegexpMatcher)
    exclusion_regexps=[r"\bavec\s+la\s+famille\b"],
    # The regexp will be used with a case-insensitivity flag
    case_sensitive=False,
    # Special chars in the input text will be converted
    # to equivalent ASCII char before runing the regexp on it
    unicode_sensitive=False,
)

family_rule_2 = FamilyDetectorRule(
    regexp=r"\bantecedents\s+familiaux\b",
    case_sensitive=False,
    unicode_sensitive=False,
)

family_detector = FamilyDetector(rules=[family_rule_1, family_rule_2], output_label="family")
# The family detector doesn't return anything but instead adds an attribute to each
# segment with a boolean value indicating if description of family history was detected or not
family_detector.run(sentence_segs)

# Print sentences detected as being related to family history
for sentence_seg in sentence_segs:
    # Retrieve the attribute created by the family detector
    family_attr = sentence_seg.attrs.get(label="family")[0]
    # Only print sentences about family history
    if family_attr.value:
        print(sentence_seg.text)
```

As with all rule-based operations, `FamilyDetector` provides
the {func}`~medkit.text.context.FamilyDetector.load_rules`
and {func}`~medkit.text.context.FamilyDetector.save_rules` methods
to facilitate their persistence to a YAML file.

## Negation detection

Detecting family history works best at the sentence level.
However, for negation and hypothesis, it is better to split sentences into smaller chunks,
as the scope of negation and hypothesis can be very limited.
For this purpose, `medkit` provides a {class}`~medkit.text.segmentation.SyntagmaTokenizer` operation.

```{code-cell} ipython3
from medkit.text.segmentation import SyntagmaTokenizer

# Here we will use the default settings of SyntagmaTokenizer,
# but you can specify your own separator patterns
syntagma_tokenizer = SyntagmaTokenizer(
    output_label="syntagma",
    # We want to keep the section and family history information
    # at the syntagma level
    attrs_to_copy=["section", "family"],
)
# The syntagma tokenizer expects sentence segments as input
syntagma_segs = syntagma_tokenizer.run(sentence_segs)

for syntagma_seg in syntagma_segs:
    print(syntagma_seg.text)
```

As you can see, a few sentences were split into smaller parts.
We can now run a {class}`~medkit.text.context.NegationDetector` instance on the syntagmata
(using the default rules).

```{code-cell} ipython3
from medkit.text.context import NegationDetector, NegationDetectorRule

# NegationDetectorRule objects have the same structure as FamilyDetectorRule
# Here we will use the default rules
negation_detector = NegationDetector(output_label="negation")
negation_detector.run(syntagma_segs)

# Display negated syntagmas
for syntagma_seg in syntagma_segs:
    negation_attr = syntagma_seg.attrs.get(label="negation")[0]
    if negation_attr.value:
        print(syntagma_seg.text)
```

## Hypothesis detection

`medkit` also provides {class}`~medkit.text.context.HypothesisDetector`,
which is very similar to {class}`~medkit.text.context.NegationDetector`,
except it also uses a list of conjugated verb forms in addition to the list of rules.
By default, verbs at conditional and future tenses indicate the presence of an hypothesis.
This can be configured alongside the list of verbs.

```{code-cell} ipython3
from medkit.text.context import HypothesisDetector

hypothesis_detector = HypothesisDetector(output_label="hypothesis")
hypothesis_detector.run(syntagma_segs)

# Display hypothesis syntagmas
for syntagma_seg in syntagma_segs:
    hypothesis_attr = syntagma_seg.attrs.get(label="hypothesis")[0]
    if hypothesis_attr.value:
        print(syntagma_seg.text)
```

As you can see, no hypothesis was detected in this document.

:::{warning}
The default settings (rules and verbs) of `HypothesisDetector` are **NOT** exhaustive
and may not yield satisfactory results.
If you plan on using `HypothesisDetector`, please consider specifying your own set of rules
and conjugated verbs that are specifically tailored to your data.
:::

## Passing context information to matched entities

Now that we have gathered all this contextual information,
we want to propagate it to the entities that we will find in the document.
This can be done using the `attrs_to_copy` mechanism that we have already seen,
which is available to all NER operations:

```{code-cell} ipython3
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher

# Create a matcher using a pretrained HuggingFace model
drbert_matcher = HFEntityMatcher(
    model="medkit/DrBERT-CASM2",
    attrs_to_copy=["section", "family", "hypothesis", "negation"],
)
# Run the matcher on the appropriate input segments
# and add the entities found back to the document
entities = drbert_matcher.run(syntagma_segs)
for entity in entities:
    doc.anns.add(entity)

# Print all entities with their contextual attributes
for entity in doc.anns.entities:
    print(entity.label, ":", entity.text)
    section_attr = entity.attrs.get(label="section")[0]
    print("section:", section_attr.value)
    family_attr = entity.attrs.get(label="family")[0]
    print("family:", family_attr.value)
    negation_attr = entity.attrs.get(label="negation")[0]
    print("negation:", negation_attr.value)
    hypothesis_attr = entity.attrs.get(label="hypothesis")[0]
    print("hypothesis:", hypothesis_attr.value)
    print()
```

Let's visualize this in context with `displacy`:

```{code-cell} ipython3
from spacy import displacy
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy

# Define a custom formatter that will also display some context flags
# ex: "disorder[fn]" for an entity with label "disorder" and
# family and negation attributes set to True
def _custom_formatter(entity):
    label = entity.label
    flags = []
    
    family_attr = entity.attrs.get(label="family")[0]
    if family_attr.value:
        flags.append("f")
    negation_attr = entity.attrs.get(label="negation")[0]
    if negation_attr.value:
        flags.append("n")
    hypothesis_attr = entity.attrs.get(label="hypothesis")[0]
    if hypothesis_attr.value:
        flags.append("h")

    if flags:
        label += "[" + "".join(flags) + "]"
    
    return label

# Pass the formatter to medkit_doc_to_displacy()
displacy_data = medkit_doc_to_displacy(doc, entity_formatter=_custom_formatter)
displacy.render(docs=displacy_data, manual=True, style="ent")
```
