---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Context Detection

In this tutorial, we will use rule-based operations to attach additional
contextual information to entities such has:
- the section in which the entity is located
- is the entity negated
- did it appear as part of a hypothesis
- it is related to the patient or it is part of their family's medical history

NB: If you are not familiar with medkit, you should probably take a look at the
[First steps](first_steps.md) tutorial before going further.

Let's start by loading a document:

```{code-cell} ipython3
from pathlib import Path
from medkit.core.text import TextDocument

doc = TextDocument.from_file(Path("../data/mtsamplesfr/1.txt"))
print(doc.text)
```

## Section detection

Medkit provides a {class}`~medkit.text.segmentation.SectionTokenizer` operation
that takes a input segments containing full document texts and splits them into
sections, returning a segment for each section.

The section tokenizer is configured with a list of trigger terms signaling the
beginning of a section, and corresponding section names. Medkit provides a
default list of possible sections
(https://github.com/medkit-lib/medkit/blob/main/medkit/text/segmentation/default_section_definition.yml)
but it is missing some sections that our document has, so we will manually
define our own section rules:

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

We have already seen sentence splitting [previously](first_steps.md) and we will
reuse the same code, with a little addition: we want the section information to
be propagated onto the sentences, ie. we want to be able to tell in which
section a sentence belongs.

For this, we will use the `attrs_to_copy` init parameter. It takes a list of
labels that we want to copy from the input segments to the new sentences
segments created by the operation. Here, we will use it to copy the "section"
attribute of the section segments (which has the section name as value):

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

In this document, we have a section dedicated to family medical history, but,
this is not always the case. To handle this, medkit provides a
{class}`~medkit.text.context.FamilyDetector`  operation based on regular
expressions. It is somewhat similar to the
{class}`~medkit.text.ner.RegexpMatcher` we have
[previously](entity_matcher.html#regular-expression-matching) seen, but instead
of returning entities, it attaches attributes to the segments it receives, with
a boolean value indicating whether it mentions family history.

Like most rule-based medkit operations, `FamilyDetector` comes with [predefined
rules](
https://github.com/medkit-lib/medkit/blob/main/medkit/text/context/family_detector_default_rules.yml)
that will be used by default if you don't provide any. For the sake of learning,
we will manually create a few rules:

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
{func}`~medkit.text.context.FamilyDetector.load_rules` and
{func}`~medkit.text.context.FamilyDetector.save_rules` methods to help you store
then in a yaml file.

## Negation detection

Detecting family history work best at the sentence level, but for negation and
hypothesis it is better to split sentences into smaller chunks, as the scope of
negation and hypothesis can be very limited. For this purpose, medkit comes with
a {class}`~medkit.text.segmentation.SyntagmaTokenizer` operation.

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

As you can see, a few sentences where split into smaller parts. We can now run a
{class}`~medkit.text.context.NegationDetector` instance on the syntagmas (using
the [default rules file](https://github.com/medkit-lib/medkit/blob/main/medkit/text/context/negation_detector_default_rules.yml)).

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

Medkit's {class}`~medkit.text.context.HypothesisDetector` is very similar to
`NegationDetector`, except that in addition to a list of rules, it also uses a
list of conjugated verb forms. By default, verbs at conditional and future
tenses will be considered to indicate the presence of an hypothesis. This can be
configured, as well as the list of verbs which is far from exhaustive.

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

```{warning}
The default settings (rules and verbs) of `HypothesisDetector` are far from
complete and may not give satisfactory results. If you plan on using
`HypothesisDetector`, you will need to come up with your own set of regexp rules
and conjugated verbs that work well for you data.
```

## Passing context information to matched entities

Now that we have gathered all this contextual information, we want to propagate
it to the entities that we will find in the document. This is easily done by
using the `attrs_to_copy` mechanism that we have already seen, and that is
available for all NER operations:

```
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

```
problem : Thrombocytose essentielle
section: head
family: False
negation: False
hypothesis: False

problem : thrombocytose essentielle
section: antecedents
family: False
negation: False
hypothesis: False

test : nombre
section: antecedents
family: False
negation: False
hypothesis: False

test : plaquettes
section: antecedents
family: False
negation: False
hypothesis: False

treatment : Hydrea
section: antecedents
family: False
negation: False
hypothesis: False

test : biopsie de moelle osseuse
section: antecedents
family: False
negation: False
hypothesis: False

problem : thrombocytose essentielle
section: antecedents
family: False
negation: False
hypothesis: False

problem : mutation JAK-2
section: antecedents
family: False
negation: False
hypothesis: False

treatment : Hydrea
section: antecedents
family: False
negation: False
hypothesis: False

problem : polyarthrite rhumatoïde
section: antecedents
family: False
negation: False
hypothesis: False

test : d
section: antecedents
family: False
negation: False
hypothesis: False

test : énergie
section: antecedents
family: False
negation: False
hypothesis: False

test : statut de performance ECOG
section: antecedents
family: False
negation: False
hypothesis: False

problem : fièvre
section: antecedents
family: False
negation: True
hypothesis: False

problem : frissons
section: antecedents
family: False
negation: True
hypothesis: False

problem : sueurs nocturnes
section: antecedents
family: False
negation: True
hypothesis: False

problem : adénopathie
section: antecedents
family: False
negation: True
hypothesis: False

problem : nausées
section: antecedents
family: False
negation: True
hypothesis: False

problem : vomissements
section: antecedents
family: False
negation: True
hypothesis: False

treatment : vitamine D
section: current_drugs
family: False
negation: False
hypothesis: False

treatment : aspirine
section: current_drugs
family: False
negation: False
hypothesis: False

treatment : vitamine C
section: current_drugs
family: False
negation: False
hypothesis: False

problem : allergie médicamenteuse
section: allergies
family: False
negation: True
hypothesis: False

test : EXAMEN DES SYSTÈMES
section: clinical_exam
family: False
negation: False
hypothesis: False

treatment : Appendicectomie
section: antecedents
family: False
negation: False
hypothesis: False

treatment : Amygdalectomie
section: antecedents
family: False
negation: False
hypothesis: False

treatment : adénoïdectomie
section: antecedents
family: False
negation: False
hypothesis: False

treatment : Chirurgie bilatérale de la cataracte
section: antecedents
family: False
negation: False
hypothesis: False

problem : tabagisme
section: life_style
family: False
negation: False
hypothesis: False

problem : tumeur solide
section: family_history
family: True
negation: False
hypothesis: False

problem : hémopathies malignes
section: family_history
family: True
negation: True
hypothesis: False

test : EXAMEN PHYSIQUE
section: clinical_exam
family: False
negation: False
hypothesis: False

problem : pèse
section: clinical_exam
family: False
negation: False
hypothesis: False
```

Let's visualize this in context with `displacy`:

```
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

<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">PLAINTE PRINCIPALE :<br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Thrombocytose essentielle<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
.<br><br>ANTÉCÉDENTS DE LA MALADIE ACTUELLE : <br>C'est un M. de 64 ans que je suis pour une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">thrombocytose essentielle<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
. Il a été initialement diagnostiqué lorsqu'il a vu un hématologue pour la première fois le 09/07/07. A cette époque, son 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">nombre<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
 de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">plaquettes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
 était de 1 240 000. Il a d'abord commencé à prendre de l'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Hydrea<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 1000 mg par jour. Le 07/11/07, il a subi une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">biopsie de moelle osseuse<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
, qui a montré une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">thrombocytose essentielle<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
. Il était positif pour la 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">mutation JAK-2<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
. Le 11/06/07, ses plaquettes étaient à 766 000. Sa dose actuelle d'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Hydrea<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 est maintenant de 1500 mg les lundis et vendredis et de 1000 mg tous les autres jours. Il a déménagé à ABCD en décembre 2009 pour tenter d'améliorer la 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">polyarthrite rhumatoïde<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
 de sa femme. Dans l'ensemble, il se porte bien. Il a un bon niveau 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">d<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">énergie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
 et son 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">statut de performance ECOG<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
 est de 0. Absence de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">fièvre<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[n]</span></mark>
, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">frissons<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[n]</span></mark>
 ou 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">sueurs nocturnes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[n]</span></mark>
. Pas d'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">adénopathie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[n]</span></mark>
. Pas de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">nausées<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[n]</span></mark>
 ni de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vomissements<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[n]</span></mark>
. Aucun changement dans les habitudes intestinales ou vésicales.<br><br>MÉDICAMENTS ACTUELS : <br>Hydrea 1500 mg les lundis et vendredis et 1000 mg les autres jours de la semaine, Mecir 1cp/j, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vitamine D<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 1/j, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">aspirine<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 80 mg 1/j et 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vitamine C<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 1/j <br><br>ALLERGIES : <br>Aucune 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">allergie médicamenteuse<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[n]</span></mark>
 connue.<br><br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">EXAMEN DES SYSTÈMES<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
 <br>Correspondant à l'histoire de la maladie. Pas d'autre signes.<br><br>ANTÉCÉDENTS MÉDICAUX :<br>1. 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Appendicectomie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
.<br>2. 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Amygdalectomie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 et une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">adénoïdectomie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
.<br>3. 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Chirurgie bilatérale de la cataracte<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
.<br>4. HTA.<br><br>MODE DE VIE : <br>Il a des antécédents de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">tabagisme<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
 qu'il a arrêté à l'âge de 37 ans. Il consomme une boisson alcoolisée par jour. Il est marié. Il est directeur de laboratoire à la retraite.<br><br>ANTÉCÉDENTS FAMILIAUX : <br>Antécédents de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">tumeur solide<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[f]</span></mark>
 dans sa famille mais aucun d'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">hémopathies malignes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem[fn]</span></mark>
.<br><br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">EXAMEN PHYSIQUE<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
 :<br>Le patient 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">pèse<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
 85.7 kg.<br></div></span>

## Adding context attributes a posteriori

What if we already have some entities that we imported from another source and
we want to attach the contextual information that we obtain with medkit
operations? In that case it is possible to use the
{class}`~medkit.text.postprocessing.AttributeDuplicator` operation, that makes
it possible to copy attributes a posteriori without using the `attrs_to_copy`
parameter.

## Wrapping it up

In this tutorial, we have seen how medkit can help you to detect contextual
information with built-in rule-based detectors, for which the rules can be
customized.

These detectors can be run on segments of different granularity,
such as sentences or syntagmas, and the results are stored in attributes.

In order to make these contextual attributes propagate from the outer-most
segments down to the entities matched, we use the `attrs_to_copy` operation
init parameter.
