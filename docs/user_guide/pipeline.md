---
jupytext:
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.13.8
kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---
# Using pipelines

This tutorial will show you how to encapsulate operations into a pipeline,
and how to create pipelines to augment documents.

## Using operations without a pipeline

Let's start by instantiating the preprocessing, segmentation, context detection
and entity recognition operations that we want to use. We are simply going to
reuse the ones from the [First steps](first_steps.md) tutorial:

```{code-cell} ipython3
from medkit.text.preprocessing import RegexpReplacer
from medkit.text.segmentation import SentenceTokenizer, SyntagmaTokenizer
from medkit.text.context import NegationDetector, NegationDetectorRule
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule

# preprocessing
rule = (r"(?<=\d)\.(?=\d)", ",")
regexp_replacer = RegexpReplacer(output_label="clean_text", rules=[rule])

# segmentation
sent_tokenizer = SentenceTokenizer(
    output_label="sentence",
    punct_chars=[".", "?", "!", "\n"],
)

synt_tokenizer = SyntagmaTokenizer(
    output_label="syntagma",
    separators=[r"\bmais\b", r"\bet\b"],
)

# context detection 
neg_rules = [
    NegationDetectorRule(regexp=r"\bpas\s*d[' e]\b"),
    NegationDetectorRule(regexp=r"\bsans\b", exclusion_regexps=[r"\bsans\s*doute\b"]),
    NegationDetectorRule(regexp=r"\bne\s*semble\s*pas"),
]
neg_detector = NegationDetector(output_label="is_negated", rules=neg_rules)

# entity recognition
regexp_rules = [
    RegexpMatcherRule(regexp=r"\ballergies?\b", label="problem"),
    RegexpMatcherRule(regexp=r"\basthme\b", label="problem"),
    RegexpMatcherRule(regexp=r"\ballegra?\b", label="treatment", case_sensitive=False),
    RegexpMatcherRule(regexp=r"\bvaporisateurs?\b", label="treatment"),
    RegexpMatcherRule(regexp=r"\bloratadine?\b", label="treatment", case_sensitive=False),
    RegexpMatcherRule(regexp=r"\bnasonex?\b", label="treatment", case_sensitive=False),
]
regexp_matcher = RegexpMatcher(rules=regexp_rules, attrs_to_copy=["is_negated"])
```

Each of these operations has a `run()` method, which we could call sequentially,
passing along the output from one operation as the input to the next operation,
and using a document's raw text segment as the initial input:


```{code-cell} ipython3
from pathlib import Path
from medkit.core.text import TextDocument

# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/text/1.txt
# or create your file and copy the text
doc = TextDocument.from_file(Path("../data/text/1.txt"))

# clean_segments contains only 1 segment: the preprocessed full text segment
clean_segments = regexp_replacer.run([doc.raw_segment])
sentences = sent_tokenizer.run(clean_segments)
syntagmas = synt_tokenizer.run(sentences)
# the negation detector doesn't return new annotations
# but rather appends attributes to the segments it received
neg_detector.run(syntagmas)
entities = regexp_matcher.run(syntagmas)
```

But it is also possible to wrap all this operations into a `Pipeline` object,
that will be responsible of calling the `run()` method of each operation, with
the appropriate input annotations.

## Why use a pipeline?

What are the advantages of using pipelines instead of just directly calling each
operations as we just did?

In this particular case, they aren't any real advantages. Because this is a
tutorial and we want to keep things simple, there aren't so many operations and
they are called in a linear fashion. But in real life the chaining of operations
could be more complex and then it could be easier to do that through a pipeline.

Also, pipelines are composable (each pipeline is an operation that can itself be
put into another pipeline), therefore they can be used to structure complex
flows into smaller units handling a subpart of the processing. This also makes
it possible to reuse a pipeline for different projects, for instance by
regrouping common preprocessing steps.

If you are interested in [provenance tracing](provenance.md) (knowing how each
annotation was generated), then it can also be easier to handle that with a
pipeline.

Finally, in the future of medkit the scope of pipelines might be expanded to
handle more things such as batching, parallelization, and maybe training of
trainable components.

## Constructing a pipeline

We now want to connect these 4 operations together in a pipeline. For this, we
will stack all the operations in a python list, in the order in which they must
be executed. But we also need to "connect" the operations together, ie. to
indicate which output of an operation should be fed as input to another
operation. This why we wrap the operations in {class}`~medkit.core.PipelineStep`
objects:

```{code-cell} ipython3
from medkit.core import PipelineStep

steps = [
    PipelineStep(regexp_replacer, input_keys=["full_text"], output_keys=["clean_text"]),
    PipelineStep(sent_tokenizer, input_keys=["clean_text"], output_keys=["sentences"]),
    PipelineStep(synt_tokenizer, input_keys=["sentences"], output_keys=["syntagmas"]),
    PipelineStep(neg_detector, input_keys=["syntagmas"], output_keys=[]),  # no output
    PipelineStep(regexp_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
]
```

Each `PipelineStep` associates an operation with “keys”. As we just said, the
operations have to be connected to each other, and the keys are just names we
put on these connections to make it easier to describe them. The steps we just
constructed can be represented like this:

```{mermaid}
:align: center
graph TD
    A((?))
    B(regexp_replacer)
    C(sent_tokenizer)
    D(synt_tokenizer)
    E(neg_detector)
    F(entity_matcher)
    G((?)):::io

    A -- full_text --> B
    B -- clean_text --> C
    C -- sentences --> D
    D -- syntagmas --> E
    E ~~~ F
    D -- syntagmas --> F
    F -- entities --> G

    classDef io fill:#fff4dd,stroke:#edb:
```

We see the negation detector has no output: this is because it modifies the
sentences in-place by adding attributes to them (its `run()` function doesn't
return anything).

The 1st question mark, connected to the sentence tokenizer via the "full_text" key, represents the source of the segments that will be fed into the regexp replacer, still unknown at this point since they are not the product of a previous operation.

The 2d question mark, connected to the entity matcher via the "entities" key, represents the destination of the entities produced by the matcher, also still unknown for now.

We will now use our pipeline steps to create a `Pipeline` object:

```{code-cell} ipython3
from medkit.core import Pipeline

pipeline = Pipeline(
    # Pipeline steps connecting the operations together
    steps,
    # Tells the pipeline that the first (and unique) argument passed to its run() method
    # corresponds to the "full_text" key
    # (and therefore should be fed as input to the regexp replacer)
    input_keys=["full_text"],
    # Tells the pipeline that the first (and unique) return value of its run() method
    # corresponds to the "entities" key
    # (and therefore that it should be the output of the regexp matcher)
    output_keys=["entities"]
)
```
Here our pipeline is the equivalent of some operation that would take full text
segments as input and return entities with family attributes. This pipeline only
has one input and one output, but creating more complex pipelines with multiple
input arguments and multiple return values is supported.

Let's run our pipeline and make sure everything is ok:

```{code-cell} ipython3
# run() takes a full text segment and return entities with attributes
entities = pipeline.run([doc.raw_segment])

for entity in entities:
    neg_attr = entity.attrs.get(label="is_negated")[0]
    print(f"text='{entity.text}', label={entity.label}, is_negated={neg_attr.value}")
```

Seems good!

## Nesting pipelines

Because a pipeline is a medkit operation (it has a `run()` method that takes
input data and return new data), it can itself be used as the step of another
pipeline. We can use this to regroup together our regexp replacer, sentence
tokenizer and family detector into a context subpipeline:

```{code-cell} ipython3
# Context pipeline that receives full text segments
# and returns preprocessed syntagmas segments with negation attributes
context_pipeline = Pipeline(
    # Optional name to indicate task performed by a pipeline
    # (will be used in provenance data)
    name="context",
    steps=[
        PipelineStep(regexp_replacer, input_keys=["full_text"], output_keys=["clean_text"]),
        PipelineStep(sent_tokenizer, input_keys=["clean_text"], output_keys=["sentences"]),
        PipelineStep(synt_tokenizer, input_keys=["sentences"], output_keys=["syntagmas"]),
        PipelineStep(neg_detector, input_keys=["syntagmas"], output_keys=[]),
    ],
    input_keys=["full_text"],
    output_keys=["syntagmas"],
)
```
Likewise, we can add an additional UMLS-based matching operation (see also
[Entity Matching](entity_matching.md)) and group it with our previous regexp
matcher into an NER subpipeline:

```{code-cell} ipython3
:tags: [skip-execution]

from medkit.text.ner import UMLSMatcher

umls_matcher = UMLSMatcher(
    umls_dir="../data/umls/2021AB/META/",
    language="FRE",
    cache_dir=".umls_cache/",
    attrs_to_copy=["is_negated"],
)

# NER pipeline that receives syntagmas segments and return entities
# matched by 2 different operations
ner_pipeline = Pipeline(
    name="ner",
    steps=[
        PipelineStep(regexp_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
        PipelineStep(umls_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
    ],
    input_keys=["syntagmas"],
    output_keys=["entities"],
)
```

Here, the 2 pipeline steps have the same output key so the pipeline's `run()`
method will return a list containing the entities matched by the regexp matcher
and the UMLS matcher.

These 2 sub-pipelines can now be grouped into an main pipeline and connected together:

```{code-cell} ipython3
:tags: [skip-execution]

pipeline = Pipeline(
    steps=[
        PipelineStep(context_pipeline, input_keys=["full_text"], output_keys=["syntagmas"]),
        PipelineStep(ner_pipeline, input_keys=["syntagmas"], output_keys=["entities"]),
    ],
    input_keys=["full_text"],
    output_keys=["entities"],
)
```

which can be represented like this:

```{mermaid}
:align: center
graph TD

    subgraph " "
    A((?))
    B(regexp_replacer)
    C(sent_tokenizer)
    D(synt_tokenizer)
    E(neg_detector)
    F((?)):::io

    A -- full_text --> B
    B -- clean_text --> C
    C -- sentences --> D
    D -- syntagmas --> E
    E ~~~ F
    D -- syntagmas --> F

    end


    subgraph " "
    G((?))
    H(regexp_matcher)
    I(umls_matcher)
    J((?)):::io

    G -- syntagmas --> H
    G -- syntagmas --> I
    H -- entities --> J
    I -- entities --> J

    end

    K((?))
    K -- full_text--> A
    F -- syntagmas --> G

    L((?))
    J -- entities --> L

    classDef io fill:#fff4dd,stroke:#edb:
```

Let's run the pipeline and make sure we still get entities with negation
attributes:

```{code-cell} ipython3
:tags: [skip-execution]

entities = pipeline.run([doc.raw_segment])

for entity in entities:
    neg_attr = entity.attrs.get(label="is_negated")[0]
    print(entity.label, ":", entity.text)
    print("negation:", neg_attr.value, end="\n\n")
```

```
problem : allergies
negation: False

problem : allergies
negation: False

treatment : Allegra
negation: False

treatment : vaporisateurs
negation: False

treatment : vaporisateurs
negation: True

problem : asthme
negation: False

problem : asthme
negation: False

treatment : Allegra
negation: False

problem : allergies
negation: True

treatment : Allegra
negation: False

treatment : loratadine
negation: False

treatment : Nasonex
negation: False

disorder : asthme
negation: False

chemical : médicaments
negation: False

disorder : asthme
negation: False

chemical : MÉDICAMENTS
negation: False

procedure : EXAMEN PHYSIQUE
negation: False

physiology : Poids
negation: False

physiology : pression sanguine
negation: False

anatomy : Yeux
negation: True

anatomy : Nez
negation: True

anatomy : Gorge
negation: True

anatomy : gorge
negation: True

anatomy : muqueuse
negation: False

procedure : drainage
negation: False

anatomy : Cou
negation: True

disorder : adénopathie
negation: True

anatomy : Poumons
negation: False
```

## Using a document pipeline

The pipeline we have created can be seen as an "annotation-level" pipeline. It
takes {class}`~medkit.core.text.Segment` objects as input and returns
{class}`~medkit.core.text.Entity` objects (`Segment` and `Entity` both being
subclasses of {class}`~medkit.core.text.TextAnnotation`).


As mentionned in a [previous tutorial](entity_matching.md), when dealing with a
collection of documents that we want to enrich with annotations, we need to
iterate over each document to obtain its entities rather than processing all the
documents at once:

```{code-cell} ipython3
docs = TextDocument.from_dir(Path("..data/text"))

for doc in docs:
    entities = pipeline.run([doc.raw_segment])
    for entity in entities:
        doc.anns.add(entity)
```

To handle this common use case, medkit provides a
{class}`~medkit.core.DocPipeline` class, that wraps a `Pipeline` instance and
run it on each document that it receives. This is how we would use it:

```{code-cell} ipython3
from medkit.core import DocPipeline

docs = TextDocument.from_dir(Path("..data/text"))

doc_pipeline = DocPipeline(pipeline=pipeline)
doc_pipeline.run(docs)
```

## Wrapping it up

In this tutorial, we have learnt how to instantiate a `Pipeline` and describe
how operations are connected with each others through `PipelineStep` objects. We
have also seen how sub-pipelines can be nested into other pipelines. Finally, we
have seen how to transform an annotation-level `Pipeline` into a document-level
`DocPipeline`.

If you have more questions about pipelines or wonder how to build more complex
flows, you may want to take a look at the [pipeline API
docs](api:core:pipeline). If you are interested in the advantages of pipelines
as regard provenance tracing, you may read the [provenance tracing tutorial](provenance.md).
