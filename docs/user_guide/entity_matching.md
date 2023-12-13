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

# Entity Matching

This tutorial will take you on a tour of the most common methods to perform
entity matching on text documents using medkit.

NB: If you are new to medkit, you should probably take a look at the [First
steps](first_steps.md) tutorial before going further.

## Sentence splitting

Before trying to locate entities in a document, it is often necessary to split
it into sentences, either because some operations expect sentences rather than a
full document as their input, or because we will afterward perform some context
detection operation at the sentence level.

Let's start by loading a medical report to work on:

```{code-cell} ipython3
from pathlib import Path
from medkit.core.text import TextDocument

doc = TextDocument.from_file(Path("../data/mtsamplesfr/1.txt"))
print(doc.text)
```

We will now use medkit's sentence tokenizing operation to create and display
sentence segments. As seen [before](first_steps.md), the sentence tokenizer
expects a list of segments as input and will return a list of sentence segments,
and since we don't have any segments yet on our document, we use
`TextDocument.raw_segment`, which is a special segment that contains the full
unprocessed text.

```{code-cell} ipython3
from medkit.text.segmentation import SentenceTokenizer

# By default, SentenceTokenizer will use a list of punctuation chars to detect sentences.
sentence_tokenizer = SentenceTokenizer(
    # Label of the segments created and returned by the operation
    output_label="sentence",
    # Keep the punctuation character inside the sentence segments
    keep_punct=True,
    # Also split on newline chars, not just punctuation characters
    split_on_newlines=True,
)

# Pass the raw segment as input
sentence_segs = sentence_tokenizer.run([doc.raw_segment])

# Print all returned sentence segments
for sentence_seg in sentence_segs:
    print(sentence_seg.text, end="\n\n")
```

## Regular expression matching

Medkit comes with a built-in matcher that can identify entities based on regular
expressions. For a complete overview of its features, you can refer to its
{mod}`API doc<medkit.text.ner.regexp_matcher>`.

We are going to use regular expressions to match entities that cannot be
detected by a dictionary-based approach, such as age and weight indications:

```{code-cell} ipython3
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule

# Rule with simple regexps to match age and weights
# references (numerical values followed by year or kilogram units)
regexp_rule_1 = RegexpMatcherRule(
    # Label of the entities found through this rule
    label="age",
    # Simple regexp (\b is word boundary)
    regexp=r"\b\d+\s+(ans?|annees?)\b",
    # The regexp will be used with a case-insensitivity flag
    case_sensitive=False,
    # Special chars in the input text will be converted
    # to equivalent ASCII char before running the regexp on it
    unicode_sensitive=False,
)

# Rule to match weight indications
# (any number followed by "kg", "kilos", "kilogrammes")
regexp_rule_2 = RegexpMatcherRule(
    label="weight",
    regexp=r"\b\d+([\.,]\d+)?\s*(kg|kilos?|kilogrammes?)\b",
    case_sensitive=False,
    unicode_sensitive=False,
)

# Create a RegexpMatcher with the rules we just defined
regexp_matcher = RegexpMatcher(rules=[regexp_rule_1, regexp_rule_2])

# The regexp matcher run() method takes a list of segments as input
# and returns a list of entities
entities = regexp_matcher.run(sentence_segs)
for entity in entities:
    print(entity.text, entity.label)
```

Let's visualize them with `displacy`, using the
{func}`~medkit.text.spacy.displacy_utils.entities_to_displacy` helper (similar
to {func}`~medkit.text.spacy.displacy_utils.medkit_doc_to_displacy` but we can
pass it a list of entities rather than a `TextDocument`):

```{code-cell} ipython3
from spacy import displacy
from medkit.text.spacy.displacy_utils import entities_to_displacy

displacy_data = entities_to_displacy(entities, doc.text)
displacy.render(displacy_data, manual=True, style="ent")
```

 Note that you can save a particular list of regexp rules into a yaml file using
 the {func}`~medkit.text.ner.RegexpMatcher.save_rules` static method, and
 then reload them with {func}`~medkit.text.ner.RegexpMatcher.load_rules`.
 This makes it easier to share and reuse them:

```{code-cell} ipython3
RegexpMatcher.save_rules([regexp_rule_1, regexp_rule_2], "weight_and_age_rules.yml")
rules = RegexpMatcher.load_rules("weight_and_age_rules.yml")
```

Medkit itself comes with a list of predefined regexp rules, available at
https://github.com/medkit-lib/medkit/blob/main/medkit/text/ner/regexp_matcher_default_rules.yml,
that will be used by default if you don't provide any rules to `RegexpMatcher`.

## Similarity-based entity matching

We will now perform entity matching but this time based on a list of terms that
we want to retrieve.

The medical report we have loaded mentions several drugs that we are interested
in detecting. For this, we are going to take a CSV file that contains commercial
names of drugs, along with the molecules they contain and their corresponding
identifiers in the ATC
(https://www.who.int/tools/atc-ddd-toolkit/atc-classification)
classification.[^atc_footnote] Let's take a look at it:

[^atc_footnote]: This file was created by Bastien Rance, reusing scripts originally from
Sébastien Cossin

```{code-cell} ipython3
import pandas as pd

drugs = pd.read_csv("../data/bdpm.csv")
drugs.head(n=10)
```

Rather than regular expressions, we will used similarity-based matching using the {class}`~medkit.text.ner.SimstringMatcher` operation.

This "fuzzy" matcher based on the [simstring algorithm](http://chokkan.org/software/simstring/) will be more tolerant to small spelling errors than the exact matching of a regular expression.We are going to create a rule for each commercial name, and to each rule we will attach the ATC identifier of each molecule when we know them:

```{code-cell} ipython3
from medkit.text.ner import SimstringMatcher, SimstringMatcherRule, SimstringMatcherNormalization

simstring_rules = []
for drug_name, rows in drugs.groupby("BN"):
    # Get all unique ATC ids for drug
    atc_ids = rows["atc"].dropna().unique()
    # Create SimstringMatcherNormalization objects
    # for all ATC ids of this drug
    norms = []
    for atc_id in atc_ids:
        norm = SimstringMatcherNormalization(kb_name="ATC", kb_id=atc_id)
        norms.append(norm)

    # Create a rule using the drug commercial name as term to match
    # and ACT ids as normalizations
    rule = SimstringMatcherRule(
        # Label of the entities found through this rule
        label="medication",
        # String to match
        term=drug_name,
        # Convert term and input text to lowercase before looking for matches
        case_sensitive=False,
        # Info about normalization attributes to create for entities
        # found through this rule
        normalizations=norms,
    )
    simstring_rules.append(rule)

# Create a simstring matcher with the rules we just defined
simstring_matcher = SimstringMatcher(
    # Rules to use
    rules=simstring_rules,
    # Minimum similarity for a match to be found
    threshold=0.9,
    # Similarity metric to use
    similarity="jaccard",
)

# Run it on the sentences
entities = simstring_matcher.run(sentence_segs)

# Print entities matched and their normalization attributes
for entity in entities:
    print(entity.label, ":", entity.text)
    for norm_attr in entity.attrs.norms:
        print(norm_attr.kb_name, norm_attr.kb_id)
    print()
```

## Advanced entity matching with IAMSystem

[IAMSystem](https://iamsystem-python.readthedocs.io/en/latest/) is an advanced
entity matcher developed by Sébastien Cossin.[^footnote_iam] It allows for a fine control of
the matching strategy and should be relatively fast, even when the dictionary of
terms to match is very large.

[^footnote_iam]: Cossin S, Jouhet V, Mougin F, Diallo G, Thiessard F. IAM at CLEF eHealth 2018: Concept Annotation and Coding in French Death Certificates. https://arxiv.org/abs/1807.03674

Let's see how to use it to match a couple of manually-defined terms:

```{code-cell} ipython3
from iamsystem import Matcher, ESpellWiseAlgo
from medkit.text.ner.iamsystem_matcher import IAMSystemMatcher

# Init an IAMSystem Matcher object
# by passing it keywords to match and other optional params
matcher = Matcher.build(
    keywords=["thrombocytose", "biopsie"],
    # Fuzzy algorithm(s) to use
    spellwise=[dict(measure=ESpellWiseAlgo.LEVENSHTEIN, max_distance=1, min_nb_char=5)],
    # Optional stopwords
    stopwords=["et"],
    # Many more optional are available, cf the IAMSystem documentation
)

# Wrap the IAMSystem object into the IAMSystemMatcher medkit operation
# so we can run it on the sentences
iam_system_matcher = IAMSystemMatcher(matcher=matcher)
entities = iam_system_matcher.run(sentence_segs)

for entity in entities:
    print(entity.label, ":", entity.text)
```

To learn more about the possibilities of `IAMSystem`, refer to its
[documentation](https://iamsystem-python.readthedocs.io/en/).

## Finding UMLS concepts

Rather than manually building a dictionary of terms to match, we may be
interested in exploiting the terms referenced by the [UMLS
metathesaurus](https://www.nlm.nih.gov/research/umls/).

Among other things, the UMLS contains a list of medical terms in different
languages, associated with a unique identifier (named CUI) for each concept they
refer to. The concepts are grouped together into "semantic types", themselves
grouped into wider groups caller "semantic groups" such as "ANAT", "CHEM",
"DISO", "PHYSIO", "PROC", etc (cf
https://lhncbc.nlm.nih.gov/semanticnetwork/download/sg_archive/SemGroups-v04.txt
for the complete list).

Medkit provides a similarity-based fuzzy matcher dedicated to the UMLS. It uses
2 files from the standard UMLS distribution : `MRSTY.RRF`, which contains all
the UMLS concepts with their CUI in a CSV-like format, and `MRCONSO.RRF` which
contains a list of terms in different languages with corresponding CUI. The
{class}`~medkit.text.ner.umls_matcher.UMLSMatcher` operation simply uses this
lists to build a dictionary of terms to match (it does not take advantage of the
hierarchical nature of UMLS concepts).

Note that the UMLS files are not freely distributable nor usable, to download
them and use you must request a license on the [UMLS
website](https://www.nlm.nih.gov/research/umls/index.html)

```
from medkit.text.ner import UMLSMatcher

# Codes of UMLS semantic groups to take into account
# (entities not belonging to these semgroups will be ignored)
umls_semgroups = [
    "ANAT",  # anatomy
    "CHEM",  # chemical
    "DEVI",  # device
    "DISO",  # disorder
    "GEOG",  # geographic
    "LIVB",  # living being
    "OBJC",  # object
    "PHEN",  # concept
    "PHYS",  # physiological
    "PROC",  # procedure
]

umls_matcher = UMLSMatcher(
    # Directory containing the UMLS files with all the terms and concepts
    umls_dir="../data/umls/2021AB/META/",
    # Language to use
    language="FRE",
    # Where to store the internal terms database of the matcher
    cache_dir=".umls_cache/",
    # Semantic groups to consider
    semgroups=umls_semgroups,
    # Don't be case-sensitive
    lowercase=True,
    # Convert special chars to ASCII before matching
    # (same a unicode_sensitive=False for regexp rules)
    normalize_unicode=True,
)

entities = umls_matcher.run(sentence_segs)

# Define custom formatter helper that will also display the CUI
def custom_formatter(entity):
    label = entity.label
    cui = entity.attrs.norms[0].kb_id
    return label + " (" + cui + ")"

displacy_data = entities_to_displacy(entities, doc.text, entity_formatter=custom_formatter)
displacy.render(displacy_data, manual=True, style="ent")
```

<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">PLAINTE PRINCIPALE :<br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Thrombocytose essentielle<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0040028)</span></mark>
.<br><br>ANTÉCÉDENTS DE LA 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">MALADIE<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0012634)</span></mark>
 ACTUELLE : <br>C'est un M. de 64 ans que je suis pour une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">thrombocytose essentielle<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0040028)</span></mark>
. Il a été initialement diagnostiqué lorsqu'il a vu un hématologue pour la première fois le 09/07/07. A cette époque, son nombre de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">plaquettes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">anatomy (C0005821)</span></mark>
 était de 1 240 000. Il a d'abord commencé à prendre de l'Hydrea 1000 mg par jour. Le 07/11/07, il a subi une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">biopsie de moelle osseuse<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">procedure (C0005954)</span></mark>
, qui a montré une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">thrombocytose essentielle<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0040028)</span></mark>
. Il était positif pour la 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">mutation JAK-2<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C3267069)</span></mark>
. Le 11/06/07, ses 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">plaquettes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">anatomy (C0005821)</span></mark>
 étaient à 766 000. Sa dose actuelle d'Hydrea est maintenant de 1500 mg les lundis et vendredis et de 1000 mg tous les autres jours. Il a déménagé à ABCD en décembre 2009 pour tenter d'améliorer la 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">polyarthrite rhumatoïde<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0003873)</span></mark>
 de sa 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">femme<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">living_being (C0043210)</span></mark>
. Dans l'ensemble, il se porte bien. Il a un bon niveau d'énergie et son statut de performance 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">ECOG<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">procedure (C0430797)</span></mark>
 est de 0. Absence de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">fièvre<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0015967)</span></mark>
, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">frissons<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0085593)</span></mark>
 ou 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">sueurs nocturnes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0028081)</span></mark>
. Pas d'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">adénopathie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0497156)</span></mark>
. Pas de nausées ni de vomissements. Aucun changement dans les habitudes intestinales ou vésicales.<br><br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">MÉDICAMENTS<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical (C0013227)</span></mark>
 ACTUELS : <br>Hydrea 1500 mg les lundis et vendredis et 1000 mg les autres jours de la semaine, Mecir 1cp/j, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vitamine D<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical (C0042866)</span></mark>
 1/j, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">aspirine<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical (C0004057)</span></mark>
 80 mg 1/j et 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vitamine C<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical (C0003968)</span></mark>
 1/j <br><br>ALLERGIES : <br>Aucune 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">allergie médicamenteuse<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0013182)</span></mark>
 connue.<br><br>EXAMEN DES SYSTÈMES <br>Correspondant à l'histoire de la 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">maladie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0012634)</span></mark>
. Pas d'autre signes.<br><br>ANTÉCÉDENTS MÉDICAUX :<br>1. 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Appendicectomie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">procedure (C0003611)</span></mark>
.<br>2. 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Amygdalectomie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">procedure (C0040423)</span></mark>
 et une 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">adénoïdectomie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">procedure (C0001425)</span></mark>
.<br>3. 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Chirurgie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">procedure (C0543467)</span></mark>
 bilatérale de la 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">cataracte<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0086543)</span></mark>
.<br>4. HTA.<br><br>MODE DE VIE : <br>Il a des antécédents de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">tabagisme<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0040332)</span></mark>
 qu'il a arrêté à l'âge de 37 ans. Il consomme une boisson alcoolisée par jour. Il est 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">marié<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">living_being (C0015209)</span></mark>
. Il est directeur de laboratoire à la 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">retraite<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0035345)</span></mark>
.<br><br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">ANTÉCÉDENTS FAMILIAUX<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0241889)</span></mark>
 : <br>Antécédents de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">tumeur solide<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0280100)</span></mark>
 dans sa 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">famille<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">living_being (C0015576)</span></mark>

<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">mais<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">object (C1138842)</span></mark>
 aucun d'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">hémopathies malignes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">disorder (C0376545)</span></mark>
.<br><br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">EXAMEN PHYSIQUE<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">procedure (C0031809)</span></mark>
 :<br>Le patient pèse 85.7 kg.<br></div></span>


## Finding entities with BERT models

BERT language models are neural network using a transformer architecture,
trained on large amounts of textual data using self-supervised learning
techniques such as masked language modeling and next sentence prediction.
Additional layers can be added to BERT models to perform various NLP tasks,
including named entity recognition.

Medkit makes it possible to use BERT models for NER by wrapping the [HuggingFace
transformers library](https://huggingface.co/docs/transformers/index). This
python deep learning library specializes in reimplementing state of the art
transformers architectures, and also provides a model hub where the weights of
many pre-trained models can be found.

[DrBERT](https://drbert.univ-avignon.fr/) is a BERT model trained on french
biomedical documents, available on the HuggingFace hub at
https://huggingface.co/Dr-BERT/DrBERT-7GB. The medkit team has fine-tuned DrBERT
on an annotated version of the [CAS dataset](https://hal.science/hal-01937096)
to perform entity matching: https://huggingface.co/medkit/DrBERT-CASM2

Let's use this model with the
{class}`~medkit.text.ner.hf_entity_matcher.HFEntityMatcher` to look for entities
in our document:

```
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher

# HFEntityMatcher just needs the name of a model on the HuggingFace hub or a path to a local checkpoint
# Here we use the pre-trained model at https://huggingface.co/medkit/DrBERT-CASM2
bert_matcher = HFEntityMatcher(model="medkit/DrBERT-CASM2")

# We run the matcher on all detected sentences
# Note that we must not call matcher.run() directly on the full text (doc.raw_segment)
# because BERT models have a limit on the length of the text that they can process
# and the full text might exceed it
entities = bert_matcher.run(sentence_segs)

displacy_data = entities_to_displacy(entities, doc.text)
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
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">fièvre<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">frissons<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
 ou 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">sueurs nocturnes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
. Pas d'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">adénopathie<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
. Pas de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">nausées<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
 ni de 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vomissements<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
. Aucun changement dans les habitudes intestinales ou vésicales.<br><br>MÉDICAMENTS ACTUELS : <br>Hydrea 1500 mg les lundis et vendredis et 1000 mg les autres jours de la semaine, Mecir 1cp/j, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vitamine D<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 1/j, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">aspirine<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 80 mg 1/j et 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">vitamine C<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">treatment</span></mark>
 1/j <br><br>ALLERGIES : <br>Aucune 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">allergie médicamenteuse<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
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
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">tumeur solide<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
 dans sa famille mais aucun d'
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">hémopathies malignes<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
.<br><br>
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">EXAMEN PHYSIQUE<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">test</span></mark>
 :<br>Le patient 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">pèse<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">problem</span></mark>
 85.7 kg.<br></div></span>


Note that the entities obtained with `HFEntityMatcher` don't have any
normalization attributes attached to them.

## Matching entities in multiple documents

Let's consider the more realistic case in which we are dealing with a collection
of documents rather than a unique document:

```{code-cell} ipython3
from glob import glob

# Let's load all of our sample documents
docs = TextDocument.from_dir(Path("../data/mtsamplesfr/"))
print(len(docs))
```

It is possible to run the sentence splitting and entity matching operations on all documents at once:

```{code-cell} ipython3
sentence_segs = sentence_tokenizer.run([d.raw_segment for d in docs])
entities = regexp_matcher.run(sentence_segs)
for entity in entities:
    print(entity.label, entity.text)
```

Here, `entities` contains the entities found by the regexp matcher across
all of our documents, in a list. But if we want to attach the entities back to
the document they belong to, then we need to process each document
independently:

```{code-cell} ipython3
for doc in docs:
    clean_text_segs = sentence_tokenizer.run([doc.raw_segment])
    sentence_segs = sentence_tokenizer.run(clean_text_segs)
    entities = regexp_matcher.run(sentence_segs)
    for entity in entities:
        doc.anns.add(entity)
```

When using pipelines (which will be covered in a later tutorial), this last use
case is covered by the {class}`~medkit.core.DocPipeline` class.

## Wrapping it up

Medkit provides many operations to perform entity matching using various
methods: regular expressions, fuzzy matching, BERT models, etc.

Even if you do complex pre-processing, medkit will be able to give the
characters pans of the entities in the original unprocessed text.

If you use different methods or 3d-party tools, it is possible to wrap them into
a medkit operation so you can use them within medkit, as described in [this
tutorial](module.md). Contributions to medkit are welcome so you can
submit your operations to be integrated into medkit!


```{code-cell} ipython3
:tags: [remove-cell]
import os

os.unlink("weight_and_age_rules.yml")
```
