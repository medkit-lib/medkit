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

# First steps

This tutorial will show you how to use medkit to annotate a text document, by
applying pre-processing, entity matching and context detections operations.

## Loading a text document

For starters, let's load a text file using the
{class}`~medkit.core.text.TextDocument` class:

```{code-cell} ipython3
# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/text/1.txt

from pathlib import Path
from medkit.core.text import TextDocument

doc = TextDocument.from_file(Path("../data/text/1.txt"))
```

The full raw text can be accessed through the `text` attribute:

```{code-cell} ipython3
print(doc.text)
```

A `TextDocument` can store {class}`~medkit.core.text.TextAnnotation` objects but
for now, our document is empty.

## Splitting a document in sentences

A common task in natural language processing is to split (or tokenize) text
documents in sentences. Medkit provides several segmentation operations,
including a rule-based {class}`~medkit.text.segmentation.SentenceTokenizer`
class that relies on a list of punctuation characters. Let's instantiate it:

```{code-cell} ipython3
from medkit.text.segmentation import SentenceTokenizer

sent_tokenizer = SentenceTokenizer(
    output_label="sentence",
    punct_chars=[".", "?", "!"],
)
```

As all operations, `SentenceTokenizer` defines a `run()` method. This method
returns a list of {class}`~medkit.core.text.Segment` objects (a `Segment` is a
`TextAnnotation` that represents a portion of a document's full raw text). As
input, it also expects a list of `Segment` objects. Here, we can pass a special
segment containing the whole raw text of the document, that we can retrieve
through the `raw_segment` attribute of `TextDocument`:

```{code-cell} ipython3
sentences = sent_tokenizer.run([doc.raw_segment])
for sentence in sentences:
    print(f"uid={sentence.uid}")
    print(f"text={sentence.text!r}")
    print(f"spans={sentence.spans}, label={sentence.label}\n")
```

As you can see, each segment has:
 - an `uid` attribute, which unique value is automatically generated;
 - a `text` attribute holding the text that the segment refers to;
 - a `spans` attribute reflecting the position of this text in the document's
   full raw text. Here we only have one span for each segment, but multiple
   discontinuous spans are supported;
 - and a `label`, always equal to `"sentence"` in our case but it could be
   different for other kinds of segments.

## Preprocessing a document

If you take a look at the 13th and 14th detected sentences, you will notice something
strange:

```{code-cell} ipython3
print(repr(sentences[12].text))
print(repr(sentences[13].text))
```

This is actually one sentence that was split into two segments, because the
sentence tokenizer incorrectly considers the dot in the decimal weight value to
mark the end of a sentence. We could be a little smarter when configuring the
tokenizer, but instead, for the sake of learning, let's fix this with a
pre-processing step that replaces dots by commas in decimal numbers.

For this, we can use the
{class}`~medkit.text.preprocessing.RegexpReplacer` class, a regexp-based
"search-and-replace" operation. As many medkit operations, it can be configured
with a set of user-determined rules:

```{code-cell} ipython3
from medkit.text.preprocessing import RegexpReplacer

rule = (r"(?<=\d)\.(?=\d)", ",") # => (pattern to replace, new text)
regexp_replacer = RegexpReplacer(output_label="clean_text", rules=[rule])
```

The `run()` method of the normalizer takes a list of `Segment` objects and
returns a list of new `Segment` objects, one for each input `Segment`. In our
case we only want to preprocess the full raw text segment and we will only
receive one preprocessed segment, so we can call it with:

```{code-cell} ipython3
clean_segment = regexp_replacer.run([doc.raw_segment])[0]
print(clean_segment.text)
```

And then we may use again our previously-defined sentence tokenizer, but this
time on the preprocessed text:

```{code-cell} ipython3
sentences = sent_tokenizer.run([clean_segment])
print(sentences[12].text)
```

Problem fixed!

## Finding entities

the medkit library also comes with operations to perform NER (named entity recognition), for
instance {class}`~medkit.text.ner.regexp_matcher.RegexpMatcher`. Let's
instantiate one with a few simple rules:

```{code-cell} ipython3
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule

regexp_rules = [
    RegexpMatcherRule(regexp=r"\ballergies?\b", label="problem"),
    RegexpMatcherRule(regexp=r"\basthme\b", label="problem"),
    RegexpMatcherRule(regexp=r"\ballegra?\b", label="treatment", case_sensitive=False),
    RegexpMatcherRule(regexp=r"\bvaporisateurs?\b", label="treatment"),
    RegexpMatcherRule(regexp=r"\bloratadine?\b", label="treatment", case_sensitive=False),
    RegexpMatcherRule(regexp=r"\bnasonex?\b", label="treatment", case_sensitive=False),
]
regexp_matcher = RegexpMatcher(rules=regexp_rules)
```

As you can see, you can also define some rules that ignore case distinctions by 
setting `case-sensitive` parameter to `False`.
In this example, we decide to make it for drugs (Allegra, Nasonex and Loratadine).

```{note}
When `RegexpMatcher` is instantiated without any rules, it will use a set of
default rules that where initially created to be used with documents in french
from the APHP EDS. These rules are stored in the
`regexp_matcher_default_rules.yml` file in the `medkit.text.ner` module.

You may also define your own rules in a `.yml` file. You can then load them
using the `RegexpMatcher.load_rules()` static method and then pass then to the
`RegexpMatcher` at init.
```

Since `RegexpMatcher` is an NER operation, its `run()` method returns a list of
{class}`~medkit.core.text.Entity` objects representing the entities that were
matched (`Entity` is a subclass of `Segment`). As input, it expects a list of
`Segment` objects. Let's give it the sentences returned by the sentence
tokenizer:

```{code-cell} ipython3
entities = regexp_matcher.run(sentences)

for entity in entities:
    print(f"uid={entity.uid}")
    print(f"text={entity.text!r}, spans={entity.spans}, label={entity.label}\n")
```

Just like sentences, each entity has `uid`, `text`, `spans` and `label` attributes (in
this case, determined by the rule that was used to match it).

## Detecting negation

So far we have detected several entities with `"problem"` or `"treatment"`
labels in our document. We might be tempted to use them directly to build a list
of problems that the patient faces and treatments that were given, but if we
look at how these entities are used in the document, we will see that some of
these entities actually denote the absence of a problem or treatment.

To solve this kind of situations, medkit comes with context detectors, such as
{class}`~medkit.text.context.negation_detector.NegationDetector`.
`NegationDetector.run()` receives a list of `Segment` objects. It doesn't return
anything but it will append an {class}`~medkit.core.Attribute` object to each
segment with a boolean value indicating whether a negation was detected or not
(`Segment` and `Entity` objects can have a list of `Attribute` objects,
accessible through their {class}`~medkit.core.AttributeContainer`).

Let's instantiate a `NegationDetector` with a couple of simplistic handcrafted
rules and run it on our sentences:

```{code-cell} ipython3
from medkit.text.context import NegationDetector, NegationDetectorRule

neg_rules = [
    NegationDetectorRule(regexp=r"\bpas\s*d[' e]\b"),
    NegationDetectorRule(regexp=r"\bsans\b", exclusion_regexps=[r"\bsans\s*doute\b"]),
    NegationDetectorRule(regexp=r"\bne\s*semble\s*pas"),
]
neg_detector = NegationDetector(output_label="is_negated", rules=neg_rules)
neg_detector.run(sentences)
```

```{note}
Similarly to `RegexpMatcher`, `DetectionDetector` also comes with a set of
default rules designed for documents from the EDS, stored in
`negation_detector_default_rules.yml` inside `medkit.text.context`.
```

And now, let's look at which sentence have been detected as being negated:

```{code-cell} ipython3
for sentence in sentences:
    neg_attr = sentence.attrs.get(label="is_negated")[0]
    if neg_attr.value:
        print(sentence.text)
```

Our simple negation detector doesn't work so bad, but sometimes
some part of the sentence has a negation and the other doesn't, and
in that case the whole sentence gets flagged as being negated.

To mitigate this, we can split each sentence into finer-grained segments called
syntagmas. Medkit provide a {class}`~medkit.text.segmentation.SyntagmaTokenizer`
for that purpose. Let's instantiate one, run it on our sentences and then run
again the negation detector but this time on the syntagmas:

```{note}
`SyntagmaTokenizer` also has default rules designed for documents from the EDS,
stored in `default_syntagma_definition.yml` inside `medkit.text.segmentation`.
```

```{code-cell} ipython3
from medkit.text.segmentation import SyntagmaTokenizer

synt_tokenizer = SyntagmaTokenizer(
    output_label="syntagma",
    separators=[r"\bmais\b", r"\bet\b"],
)
syntagmas = synt_tokenizer.run(sentences)
neg_detector.run(syntagmas)

for syntagma in syntagmas:
    neg_attr = syntagma.attrs.get(label="is_negated")[0]
    if neg_attr.value:
        print(syntagma.text)
```

That's a little better. We now have some information about negation attached to
syntagmas, but our end goal is really to know, for each entity, whether it
should be considered as negated or not. In more practical terms, we now have
negation attributes attached to our syntagmas, but what we really want is to
have negation attributes attached to entities.

In medkit, the way to do this is to use the `attrs_to_copy` parameter. This
parameter is available on all NER operations. It is used to tell the operation
which attributes should be copied from the input segments to the newly matched
entities (based on their label). In other words, it provides a way to propagate
context attributes (such as negation attributes) for segments to entities.

Let's again use a `RegexpMatcher` to find some entities, but this time from
syntagmas rather than from sentences, and using `attrs_to_copy` to copy negation
attributes:

```{code-cell} ipython3
regexp_matcher = RegexpMatcher(rules=regexp_rules, attrs_to_copy=["is_negated"])
entities = regexp_matcher.run(syntagmas)

for entity in entities:
    neg_attr = entity.attrs.get(label="is_negated")[0]
    print(f"text='{entity.text}', label={entity.label}, is_negated={neg_attr.value}")
```

We now have a negation `Attribute` for each entity!

## Augmenting a document

We now have an interesting set of annotations. We might want to process them
directly, for instance to generate table-like data about patient treatment in
order to compute some statistics. But we could also want to attach them back to
our document in order to save them or export them to some format.

The annotations of a text document can be access with `TextDocument.anns`,
an instance of {class}`~medkit.core.text.TextAnnotationContainer`) that behaves
roughly like a list but also offers additional filtering methods. Annotations
can be added by calling its `add()` method:

```{code-cell} ipython3
for entity in entities:
    doc.anns.add(entity)
```

The document and its entities can then be exported to supported external formats
(cf {class}`~medkit.io.brat.BratOutputConverter` and
{class}`~medkit.io.doccano.DoccanoOutputConverter`), or serialized in the
{mod}`~medkit.io.medkit_json` format:

```{code-cell} ipython3
from medkit.io import medkit_json

medkit_json.save_text_document(doc, "doc_1.json")
```

## Visualizing entities with displacy

Rather than printing entities, we can visualize them with `displacy`, a
visualization tool part of the [spaCy](https://spacy.io/) NLP library. Medkit
provides helper functions to facilitate the use of `displacy` in the
{mod}`~medkit.text.spacy.displacy_utils` module:

```{code-cell} ipython3
:tags: [scroll-output]

from spacy import displacy
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy

displacy_data = medkit_doc_to_displacy(doc)
displacy.render(displacy_data, manual=True, style="ent")
```

## Wrapping it up

In this tutorial, we have:
- created a `TextDocument` from an existing text file;
- instantiated several pre-processing, segmentation, context detection and
  entity matching operations;
- ran these operations sequentially over the document and obtained entities;
- attached these entities back to the original document.

The operations we have used in this tutorial are rather basic ones, mostly
rule-based, but there are many more available in medkit, including model-based
NER operations. You can learn about them in the [API reference](../api/text.md).

That's a good first overview of what you can do with medkit! To dive in further,
you might take a look at an overview of the [various entity matching methods
available in medkit](entity_matching.md), [context
detection](context_detection.md), or [how to encapsulate all these operations in
a pipeline](pipeline.md).
