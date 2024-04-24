# First steps

This tutorial will show you how to use `medkit` to annotate a text document,
by successively applying pre-processing, entity matching
and context detection operations.

## Loading a text document

For starters, let's load a text file using the {class}`~medkit.core.text.TextDocument` class:

```{code} python
# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/text/1.txt

from pathlib import Path
from medkit.core.text import TextDocument

doc = TextDocument.from_file(Path("../data/text/1.txt"))
```

The full raw text can be accessed through the `text` attribute:

```{code} python
print(doc.text)
```

A `TextDocument` can store {class}`~medkit.core.text.TextAnnotation` objects.
For now, our document is free of annotations.

## Splitting a document in sentences

A common task in natural language processing is to split (or tokenize) text
documents in sentences.

`medkit` provides several segmentation operations,
including a rule-based {class}`~medkit.text.segmentation.SentenceTokenizer` class
that relies on a list of punctuation characters.

```{code} python
from medkit.text.segmentation import SentenceTokenizer

sent_tokenizer = SentenceTokenizer(
    output_label="sentence",
    punct_chars=[".", "?", "!"],
)
```

As all operations, `SentenceTokenizer` defines a `run()` method.

This method accepts a list of {class}`~medkit.core.text.Segment` objects
(a `Segment` is a `TextAnnotation` that represents parts of a document's raw text)
and returns a list of `Segment` objects.

Here, we can pass a special `Segment` containing the full text of the document,
which can be retrieved through the `raw_segment` attribute of `TextDocument`:

```{code} python
sentences = sent_tokenizer.run([doc.raw_segment])

for sentence in sentences:
    print(f"uid={sentence.uid}")
    print(f"text={sentence.text!r}")
    print(f"spans={sentence.spans}, label={sentence.label}\n")
```

Each segment features:
 - an `uid` attribute, which unique value is automatically generated;
 - a `text` attribute holding the text that the segment refers to;
 - a `spans` attribute reflecting the position of this text in the document's raw text.
   Here, there is only one span per segment, but multiple discontinuous spans are supported;
 - a `label` attribute (set to "sentence" in our example),
   which could be different for other kinds of segments.

## Preprocessing a document

If you take a look at the 13th and 14th detected sentences,
you will notice something strange:

```{code} python
print(repr(sentences[12].text))
print(repr(sentences[13].text))
```

This is actually one sentence that was split into two segments,
because the sentence tokenizer incorrectly considers the dot in the decimal weight value
to mark the end of a sentence.
We could be a little smarter when configuring the tokenizer,
but instead, for the sake of learning,
let's fix this with a pre-processing step that replaces dots by commas in decimal numbers.

For this, we can use the {class}`~medkit.text.preprocessing.RegexpReplacer` class,
a regexp-based "search-and-replace" operation.
As other `medkit` operations, it can be configured with a set of user-determined rules:

```{code} python
from medkit.text.preprocessing import RegexpReplacer

rule = (r"(?<=\d)\.(?=\d)", ",")  # => (pattern to replace, new text)
regexp_replacer = RegexpReplacer(output_label="clean_text", rules=[rule])
```

The `run()` method of the normalizer takes a list of `Segment` objects
and returns a list of new `Segment` objects, one for each input `Segment`.
In our case we only want to preprocess the full raw text segment,
and we will only receive one preprocessed segment,
so we can call it with:

```{code} python
clean_segment = regexp_replacer.run([doc.raw_segment])[0]
print(clean_segment.text)
```

We may use again our previously-defined sentence tokenizer again,
but this time on the preprocessed text:

```{code} python
sentences = sent_tokenizer.run([clean_segment])
print(sentences[12].text)
```

Problem fixed!

## Finding entities

The `medkit` library also comes with operations to perform NER (named entity recognition),
for instance with {class}`~medkit.text.ner.regexp_matcher.RegexpMatcher`.
Let's instantiate one with a few simple rules:

```{code} python
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

As you can see, you can also define some rules that ignore case distinctions
by setting `case-sensitive` parameter to `False`.
In this example, we decide to make it for drugs (Allegra, Nasonex and Loratadine).

:::{note}
When `RegexpMatcher` is instantiated without any rules,
it will use a set of default rules that where initially created
to be used with documents in French from the APHP EDS.
These rules are stored in file `regexp_matcher_default_rules.yml`
located in the `medkit.text.ner` module.

You may also define your own rules in a `.yml` file.
You can then load them using the `RegexpMatcher.load_rules()` static method
and pass them to the `RegexpMatcher` constructor.
:::

Since `RegexpMatcher` is an NER operation,
its `run()` method returns a list of {class}`~medkit.core.text.Entity` objects
representing the entities that were matched (`Entity` is a subclass of `Segment`).
As input, it expects a list of `Segment` objects.
Let's give it the sentences returned by the sentence tokenizer:

```{code} python
entities = regexp_matcher.run(sentences)

for entity in entities:
    print(f"uid={entity.uid}")
    print(f"text={entity.text!r}, spans={entity.spans}, label={entity.label}\n")
```

Just like sentences, each entity features `uid`, `text`, `spans` and `label` attributes
(in this case, determined by the rule that was used to match it).

## Detecting negation

So far, we have detected several entities with `"problem"` or `"treatment"` labels in our document.
We might be tempted to use them directly
to build a list of problems that the patient faces and treatments that were given,
but if we look at how these entities are used in the document,
we will see that some of these entities actually denote the absence of a problem or treatment.

To solve this kind of situation, `medkit` comes with context detectors,
such as {class}`~medkit.text.context.negation_detector.NegationDetector`.
`NegationDetector.run()` receives a list of `Segment` objects.
It does not return anything, but it will append an {class}`~medkit.core.Attribute` object
to each segment with a boolean value indicating whether a negation was detected or not
(`Segment` and `Entity` objects can have a list of `Attribute` objects,
accessible through their {class}`~medkit.core.AttributeContainer`).

Let's instantiate a `NegationDetector` with a couple of simplistic handcrafted rules
and run it on our sentences:

```{code} python
from medkit.text.context import NegationDetector, NegationDetectorRule

neg_rules = [
    NegationDetectorRule(regexp=r"\bpas\s*d[' e]\b"),
    NegationDetectorRule(regexp=r"\bsans\b", exclusion_regexps=[r"\bsans\s*doute\b"]),
    NegationDetectorRule(regexp=r"\bne\s*semble\s*pas"),
]
neg_detector = NegationDetector(output_label="is_negated", rules=neg_rules)
neg_detector.run(sentences)
```

:::{note}
Similarly to `RegexpMatcher`, `DetectionDetector` also comes with a set of default rules
designed for documents from the EDS,
which are stored in file `negation_detector_default_rules.yml`
located in the `medkit.text.context` module.
:::

And now, let's look at which sentence have been detected as being negated:

```{code} python
for sentence in sentences:
    neg_attr = sentence.attrs.get(label="is_negated")[0]
    if neg_attr.value:
        print(sentence.text)
```

Our simple negation detector does not work too bad,
but sometimes some part of the sentence is tagged with a negation whilst the rest does not,
resulting in the whole sentence getting flagged as being negated.

To mitigate this, each sentence can be split into finer-grained segments called syntagmas.
`medkit` provides a {class}`~medkit.text.segmentation.SyntagmaTokenizer` for that purpose.
Let's instantiate one, apply it to our sentences and run the negation detector again,
but this time on the syntagmas:

:::{note}
`SyntagmaTokenizer` also has default rules designed for documents from the EDS,
which are stored in file `default_syntagma_definition.yml`
located in the `medkit.text.segmentation` module.
:::

```{code} python
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

We now have some information about negation attached to syntagmas,
but the end goal is really to know, for each entity,
whether it should be considered as negated or not.
In more practical terms, we have got negation attributes attached to our syntagmas,
but what we would prefer is to have negation attributes attached to entities.

In `medkit`, the way to do this is to use the `attrs_to_copy` parameter,
which is available for all NER operations.
This parameter tells the operation which attributes should be copied
from the input segments to the newly matched entities (based on their label).
In other words, it provides a way to propagate context attributes
(such as negation attributes) for segments to entities.

Let's again use a `RegexpMatcher` to find some entities,
but this time from syntagmas rather than from sentences,
and using `attrs_to_copy` to copy negation attributes:

```{code} python
regexp_matcher = RegexpMatcher(rules=regexp_rules, attrs_to_copy=["is_negated"])
entities = regexp_matcher.run(syntagmas)

for entity in entities:
    neg_attr = entity.attrs.get(label="is_negated")[0]
    print(f"text='{entity.text}', label={entity.label}, is_negated={neg_attr.value}")
```

We now have a negation `Attribute` for each entity!

## Augmenting a document

We now have an interesting set of annotations.
We might want to process them directly,
for instance to generate table-like data about patient treatment
in order to compute some statistics.
But we could also want to attach them back to our document
in order to save them or export them to some format.

The annotations of a text document can be access with `TextDocument.anns`,
an instance of {class}`~medkit.core.text.TextAnnotationContainer`)
that behaves roughly like a list but also offers additional filtering methods.
Annotations can be added by calling its `add()` method:

```{code} python
for entity in entities:
    doc.anns.add(entity)
```

The document and its corresponding entities can be exported to supported formats
such as brat (see {class}`~medkit.io.brat.BratOutputConverter`)
or Doccano (see {class}`~medkit.io.doccano.DoccanoOutputConverter`),
or serialized to JSON (see {mod}`~medkit.io.medkit_json`):

```{code} python
from medkit.io import medkit_json

medkit_json.save_text_document(doc, "doc_1.json")
```

## Visualizing entities with displacy

Rather than printing entities, we can visualize them with `displacy`,
a visualization tool part of the [spaCy](https://spacy.io/) NLP library.
`medkit` provides helper functions to facilitate the use of `displacy`
in the {mod}`~medkit.text.spacy.displacy_utils` module:

```{code} python
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
- run these operations sequentially over the document and obtained entities;
- attached these entities back to the original document.

The operations used throughout this tutorial are rather basic ones, mostly rule-based,
but there are many more available in `medkit`,
including model-based NER operations.
You can learn more about them in the [API reference](../reference/text.md).

To dive further into `medkit`, you might be interested in an overview
of the [various entity matching methods available in medkit](../tutorial/entity_matching.md),
[context detection](../tutorial/context_detection.md),
or [how to encapsulate all these operations in a pipeline](./pipeline.md).
