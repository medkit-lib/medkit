# Using pipelines

This tutorial will show you how to encapsulate operations into a pipeline,
and how to create pipelines to enrich documents.

## Using operations without a pipeline

Let's reuse the preprocessing, segmentation, context detection and entity recognition operations
from the [First steps](./first_steps.md) tutorial:

:::{code}
from medkit.text.preprocessing import RegexpReplacer
from medkit.text.segmentation import SentenceTokenizer, SyntagmaTokenizer
from medkit.text.context import NegationDetector, NegationDetectorRule
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule

# preprocessing
rule = (r"(?<=\d)\.(?=\d)", ",")
regexp_replacer = RegexpReplacer(output_label="clean_text", rules=[rule])

# segmentation
sentence_tokenizer = SentenceTokenizer(
    output_label="sentence",
    punct_chars=[".", "?", "!", "\n"],
)

syntagma_tokenizer = SyntagmaTokenizer(
    output_label="syntagma",
    separators=[r"\bmais\b", r"\bet\b"],
)

# context detection 
neg_rules = [
    NegationDetectorRule(regexp=r"\bpas\s*d[' e]\b"),
    NegationDetectorRule(regexp=r"\bsans\b", exclusion_regexps=[r"\bsans\s*doute\b"]),
    NegationDetectorRule(regexp=r"\bne\s*semble\s*pas"),
]
negation_detector = NegationDetector(
    output_label="is_negated",
    rules=neg_rules,
)

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
:::

Each of these operations features a `run()` method, which could be called sequentially.
Data need to be routed manually between inputs and outputs for each operation,
using a document's raw text segment as initial input:

:::{code}
from pathlib import Path
from medkit.core.text import TextDocument

# You can download the file available in source code
# !wget https://raw.githubusercontent.com/TeamHeka/medkit/main/docs/data/text/1.txt
# or create your file and copy the text.
doc = TextDocument.from_file(Path("../data/text/1.txt"))

# clean_segments contains only 1 segment: the preprocessed full text segment
clean_segments = regexp_replacer.run([doc.raw_segment])
sentences = sentence_tokenizer.run(clean_segments)
syntagmas = syntagma_tokenizer.run(sentences)

# Rhe negation detector does not return new annotations,
# but rather appends attributes to the segments it received.
negation_detector.run(syntagmas)
entities = regexp_matcher.run(syntagmas)
:::

This way of coding is useful for interactive exploration of `medkit`.
In the next section, we will introduce a different way using `Pipeline` objects.

## Why use a pipeline?

The previous example features a linear sequence of operations,
which is simple enough to fit on a single page of code.
More advanced use cases may require composition of more operations,
with a more complex dependency graph and more parameters to handle. 
Pipelines allows breaking an arbitrary workflow of operations
into functionally simpler and reusable units of computation. 

If you are interested in [provenance tracing](./provenance.md)
(providing metadata regarding how each annotation was generated),
then it can also be easier to handle it with a pipeline.

Planned extensions to `medkit` include support for batching
(applying a pipeline to multiple documents), parallelization,
and trainable components.

## Constructing a pipeline

We now want to compose these 4 operations together in a pipeline.
For this, we will stack all the operations in a python list,
in the order in which they must be executed.
But we also need to "connect" the operations together,
i.e. to indicate which output of an operation should be fed as input to another operation.
This is the purpose of the {class}`~medkit.core.PipelineStep` objects:

:::{code}
from medkit.core import PipelineStep

steps = [
    PipelineStep(regexp_replacer, input_keys=["full_text"], output_keys=["clean_text"]),
    PipelineStep(sentence_tokenizer, input_keys=["clean_text"], output_keys=["sentences"]),
    PipelineStep(syntagma_tokenizer, input_keys=["sentences"], output_keys=["syntagmas"]),
    PipelineStep(negation_detector, input_keys=["syntagmas"], output_keys=[]),  # no output
    PipelineStep(regexp_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
]
:::

Each `PipelineStep` associates an operation with input and output _keys_.
Pipeline steps with matching input and output keys will be connected to each other.
The resulting pipeline can be represented like this:

:::{mermaid}
---
align: center
---
graph TD
    A((full_text)):::io
    B(regexp_replacer)
    C(sentence_tokenizer)
    D(syntagma_tokenizer)
    E(negation_detector)
    F(entity_matcher)
    G((entities)):::io

    A --> B
    B -- clean_text --> C
    C -- sentences --> D
    D -- syntagmas --> E
    E ~~~ F
    D -- syntagmas --> F
    F --> G

    classDef io fill:#fff4dd,stroke:#edb:
:::

Pipeline steps can then be used to instantiate a {class}`~medkit.core.Pipeline` object:

:::{code}
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
:::

The resulting pipeline is functionally equivalent to some operation
processing full text segments as input and returning entities with family attributes as output.
This example pipeline features a single input and a single output,
but more complex pipelines with multiple inputs and outputs are supported.

Like any other operation, the pipeline can be evaluated using its `run` method: 

:::{code}
entities = pipeline.run([doc.raw_segment])

for entity in entities:
    neg_attr = entity.attrs.get(label="is_negated")[0]
    print(f"text='{entity.text}', label={entity.label}, is_negated={neg_attr.value}")
:::

## Nested pipelines

Since a pipeline is a `medkit` operation, it can be used as a step for another pipeline.
Nesting pipelines is useful to group operations into functional sub-blocks,
which can be used, tested and exercised in isolation. 

In our example, we can use this feature to regroup together our regexp replacer,
sentence tokenizer and family detector into a context sub-pipeline:

:::{code}
# Context pipeline that receives full text segments
# and returns preprocessed syntagmas segments with negation attributes.
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
:::

Likewise, we can introduce a NER sub-pipelines
composed of a UMLS-based matching operation (see also [Entity Matching](../tutorial/entity_matching.md))
grouped with the previously defined regexp matcher:

:::{code}
from medkit.text.ner import UMLSMatcher

umls_matcher = UMLSMatcher(
    umls_dir="../data/umls/2021AB/META/",
    language="FRE",
    cache_dir=".umls_cache/",
    attrs_to_copy=["is_negated"],
)

# NER pipeline that receives syntagmas segments
# and return entities matched by 2 different operations
ner_pipeline = Pipeline(
    name="ner",
    steps=[
        PipelineStep(regexp_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
        PipelineStep(umls_matcher, input_keys=["syntagmas"], output_keys=["entities"]),
    ],
    input_keys=["syntagmas"],
    output_keys=["entities"],
)
:::

Since both pipeline steps feature the same output key (_entities_),
the pipeline will return a list containing the entities matched by
both the regexp matcher and the UMLS matcher.

The NER and context sub-pipelines can now be sequenced with:

:::{code}
pipeline = Pipeline(
    steps=[
        PipelineStep(context_pipeline, input_keys=["full_text"], output_keys=["syntagmas"]),
        PipelineStep(ner_pipeline, input_keys=["syntagmas"], output_keys=["entities"]),
    ],
    input_keys=["full_text"],
    output_keys=["entities"],
)
:::

which can be represented like this:

```{mermaid}
:align: center
graph TD
    subgraph " "
    B(regexp_replacer)
    C(sentence_tokenizer)
    D(syntagma_tokenizer)
    E(negation_detector)
    B -- clean text --> C
    C -- sentences --> D
    D -- syntagmas --> E
    end

    A((full text))
    A --> B
    G((syntagmas))
    E ~~~ G
    D --> G

    subgraph " "
    H(regexp_matcher)
    I(umls_matcher)
    G --> H
    G --> I
    end

    J((entities)):::io
    H --> J
    I --> J

    classDef io fill:#fff4dd,stroke:#edb:
```

Let's run the pipeline and verify entities with negation attributes:

:::{code}
entities = pipeline.run([doc.raw_segment])

for entity in entities:
    neg_attr = entity.attrs.get(label="is_negated")[0]
    print(entity.label, ":", entity.text)
    print("negation:", neg_attr.value, end="\n\n")
:::

```text
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

The pipeline we have created can be considered an "annotation-level" pipeline.
It takes {class}`~medkit.core.text.Segment` objects as inputs
and returns {class}`~medkit.core.text.Entity` objects
(`Segment` and `Entity` both being subclasses of {class}`~medkit.core.text.TextAnnotation`).

To scale the processing of such pipeline to a collection of documents,
one needs to iterate over each document manually to obtain its entities
rather than processing all the documents at once:

:::{code}
docs = TextDocument.from_dir(Path("..data/text"))

for doc in docs:
    entities = pipeline.run([doc.raw_segment])
    for entity in entities:
        doc.anns.add(entity)
:::

To handle this common use case, `medkit` provides a {class}`~medkit.core.DocPipeline` class,
which wraps a `Pipeline` instance and run it on a list of documents.

Here is an example of its usage:

:::{code}
from medkit.core import DocPipeline

docs = TextDocument.from_dir(Path("..data/text"))

doc_pipeline = DocPipeline(pipeline=pipeline)
doc_pipeline.run(docs)
:::

## Summary

In this section, we have learnt how to instantiate a `Pipeline`
and describe how operations are connected with each others through `PipelineStep` objects.
We have also seen how sub-pipelines can be nested to compose larger pipelines.
Finally, we have seen how to transform an annotation-level pipeline
to a document-level pipeline with `DocPipeline`.
