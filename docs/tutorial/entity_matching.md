# Entity Matching

This tutorial will take you on a tour of the most common methods to perform
entity matching on text documents using `medkit`.

NB: If you are new to `medkit`, you should probably take a look at the
[First steps](../user_guide/first_steps.md) tutorial before going further.

## Sentence splitting

Before trying to locate entities in a document,
it is often necessary to split it into sentences,
either because some operations expect sentences rather than a full document as their input,
or because we will afterward perform some context detection operation at the sentence level.

Let's start by loading a medical report to work on:

```{code} python
from pathlib import Path
from medkit.core.text import TextDocument

# In case this notebook is executed outside medkit, download the example data with:
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/mtsamplesfr/1.txt
# and adjust the path below.
doc_file = Path("../data/mtsamplesfr/1.txt")
doc = TextDocument.from_file(doc_file)
print(doc.text)
```

We will now use a sentence tokenizing operation to create and display sentence segments.
As seen [before](../user_guide/first_steps.md), the sentence tokenizer expects
a list of segments as input and will return a list of sentence segments.
Since we don't have any segments yet on our document,
we use {class}`medkit.core.text.document.TextDocument`.raw_segment,
which is a special segment that contains the full unprocessed text.

```{code} python
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

`medkit` comes with a built-in matcher that can identify entities based on regular expressions.
For a complete overview of its features, you can refer to {mod}`medkit.text.ner.regexp_matcher`.

We are going to use regular expressions to match entities
that cannot be detected by a dictionary-based approach,
such as age and weight indications:

```{code} python
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

Let's visualize them with `displacy`, using {func}`~medkit.text.spacy.displacy_utils.entities_to_displacy`
(similar to {func}`~medkit.text.spacy.displacy_utils.medkit_doc_to_displacy`but we can pass it
a list of entities rather than a `TextDocument`):

```{code} python
from spacy import displacy
from medkit.text.spacy.displacy_utils import entities_to_displacy

displacy_data = entities_to_displacy(entities, doc.text)
displacy.render(displacy_data, manual=True, style="ent")
```

Note that you can save a particular list of regexp rules into a yaml file
using {func}`~medkit.text.ner.RegexpMatcher.save_rules`,
and reload them with {func}`~medkit.text.ner.RegexpMatcher.load_rules`.
This makes rules easier to share and reuse:

```{code} python
RegexpMatcher.save_rules([regexp_rule_1, regexp_rule_2], "weight_and_age_rules.yml")
rules = RegexpMatcher.load_rules("weight_and_age_rules.yml")
```

`medkit` comes with a list of predefined regexp rules,
available at https://github.com/medkit-lib/medkit/blob/main/medkit/text/ner/regexp_matcher_default_rules.yml,
which will be used as default if no rules are provided to the `RegexpMatcher` instance.

## Similarity-based entity matching

We will now perform entity matching but this time based on a list of terms
that we want to retrieve.

The medical report we have loaded mentions several drugs that we are interested in detecting.
For this, we are going to take a CSV file that contains commercial names of drugs,
along with the molecules they contain and their corresponding identifiers in the [ATC classification].[^atc_footnote]
Let's take a look at it:

[ATC classification]: https://www.who.int/tools/atc-ddd-toolkit/atc-classification
[^atc_footnote]: This file was created by Bastien Rance, reusing scripts originally from
Sébastien Cossin

```{code} python
import pandas as pd

drugs = pd.read_csv("../data/bdpm.csv")
drugs.head(n=10)
```

Rather than regular expressions, we will use similarity-based matching
using the {class}`~medkit.text.ner.SimstringMatcher` operation.

This "fuzzy" matcher based on the [simstring algorithm](http://chokkan.org/software/simstring/)
will be more tolerant to small spelling errors than the exact matching of a regular expression.
We are going to create a rule for each commercial name, and to each rule we will attach
the ATC identifier of each molecule when we know them:

```{code} python
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

[IAMSystem](https://iamsystem-python.readthedocs.io/en/latest/) is an advanced entity matcher
developed by Sébastien Cossin.[^iam_footnote]
It allows for a fine control of the matching strategy and should be relatively fast,
even when the dictionary of terms to match is very large.

[^iam_footnote]: Cossin S, Jouhet V, Mougin F, Diallo G, Thiessard F. IAM at CLEF eHealth 2018: Concept Annotation and Coding in French Death Certificates. https://arxiv.org/abs/1807.03674

Let's see how to use it to match a couple of manually-defined terms:

```{code} python
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

To learn more about the possibilities of `IAMSystem`,
please refer to its [documentation](https://iamsystem-python.readthedocs.io/en/).

## Finding UMLS concepts

Rather than manually building a dictionary of terms to match,
we may be interested in exploiting the terms referenced by the [UMLS metathesaurus].

[UMLS metathesaurus]: https://www.nlm.nih.gov/research/umls/

Among other things, the UMLS contains a list of medical terms in different languages,
associated with a unique identifier (named CUI) for each concept they refer to.
The concepts are grouped together into _semantic types_,
themselves grouped into wider groups caller [semantic groups],
such as "ANAT", "CHEM", "DISO", "PHYSIO", "PROC", etc...

[semantic groups]: https://lhncbc.nlm.nih.gov/semanticnetwork/download/sg_archive/SemGroups-v04.txt

`medkit` provides a similarity-based fuzzy matcher dedicated to the UMLS.

It uses two files from the standard UMLS distribution:
- `MRSTY.RRF` -- which contains all UMLS concepts with their CUI in a CSV-like format;
- `MRCONSO.RRF` -- which contains a list of terms in different languages with corresponding CUI.

The {class}`~medkit.text.ner.umls_matcher.UMLSMatcher` operation simply uses these lists
to build a dictionary of terms to match (it does not take advantage of the hierarchical nature of UMLS concepts).

Note that the UMLS files are not freely reusable nor redistributable nor usable.
To download them, you must request a license on the [UMLS website].

[UMLS website]: https://www.nlm.nih.gov/research/umls/index.html

```{code} python
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

## Finding entities with BERT models

BERT language models are neural network using a transformer architecture,
trained on large amounts of textual data using self-supervised learning techniques
such as masked language modeling and next sentence prediction.
Additional layers can be added to BERT models to perform various NLP tasks,
including named entity recognition.

`medkit` makes it possible to use BERT models for NER by wrapping the [HuggingFace transformers library].
This deep learning library specializes in reimplementing state-of-the-art transformers architectures,
and also provides a model hub with the weights of many pre-trained models.

[HuggingFace transformers library]: https://huggingface.co/docs/transformers/index

[DrBERT](https://drbert.univ-avignon.fr/) is a BERT model trained on french
biomedical documents, available on [HuggingFace](https://huggingface.co/Dr-BERT/DrBERT-7GB).
The medkit team fine-tuned DrBERT on an annotated version of the [CAS dataset](https://hal.science/hal-01937096)
to perform [entity matching](https://huggingface.co/medkit/DrBERT-CASM2).

Let's use this model using {class}`~medkit.text.ner.hf_entity_matcher.HFEntityMatcher`
to look for entities in our document:

```{code} python
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

Note that the entities obtained with `HFEntityMatcher` don't have any normalization attributes attached to them.

## Matching entities in multiple documents

Let's consider a more realistic case in which we are dealing with a collection of documents
rather than a unique document:

```{code} python
from glob import glob

# Let's load all of our sample documents
docs = TextDocument.from_dir(Path("../data/mtsamplesfr/"))
print(len(docs))
```

It is possible to run the sentence splitting and entity matching operations on all documents at once:

```{code} python
sentence_segs = sentence_tokenizer.run([d.raw_segment for d in docs])
entities = regexp_matcher.run(sentence_segs)
for entity in entities:
    print(entity.label, entity.text)
```

Here, `entities` contains a list of entities found by the regexp matcher across all of our documents.
But if we want to attach the entities back to the document they belong to,
then we need to process each document independently:

```{code} python
for doc in docs:
    clean_text_segs = sentence_tokenizer.run([doc.raw_segment])
    sentence_segs = sentence_tokenizer.run(clean_text_segs)
    entities = regexp_matcher.run(sentence_segs)
    for entity in entities:
        doc.anns.add(entity)
```

When using [pipelines](../user_guide/pipeline.md),
this last use case is covered using {class}`~medkit.core.DocPipeline`.

## Wrapping it up

`medkit` provides many operations to perform entity matching using various methods:
regular expressions, fuzzy matching, BERT models, etc.

Even if you do complex pre-processing, `medkit` will be able to give the characters spans
of the entities in the original unprocessed text.

If you use different methods or 3d-party tools, it is possible to wrap them into a `medkit` operation,
so you can use them anywhere else within `medkit`. See the [module](../user_guide/module.md) section.

Contributions to `medkit` are welcome, feel free to submit your operations.

```{code} python
import os

os.unlink("weight_and_age_rules.yml")
```
