# Text Components

This page contains all core text concepts of `medkit`.

For more details about public APIs, please refer to {mod}`medkit.core.text`.

```{contents} Table of Contents
:depth: 3
```

## Data Structures

The {class}`~.text.TextDocument` class implements the {class}`~.core.Document` protocol.
It allows to store subclasses of {class}`~.text.TextAnnotation`,
which implements the {class}`~.core.annotation.Annotation` protocol.

```{mermaid}
:align: center
:caption: Text document and text annotation

classDiagram
     direction TB
     class Document~Annotation~{
        <<protocol>>
    }
    class Annotation{
        <<protocol>>
    }
    class TextDocument{
        uid: str
        anns: TextAnnotationContainer
    }
    class TextAnnotation{
        <<abstract>>
        uid: str
        label: str
        attrs: AttributeContainer
    }
    Document <|.. TextDocument: implements
    Annotation <|.. TextAnnotation: implements
    TextDocument *-- TextAnnotation: contains \n(TextAnnotationContainer)
```

### Document

{class}`~.text.TextDocument` relies on {class}`~.text.TextAnnotationContainer` to manage the annotations.

Given a text document named `doc`, one can:

- browse segments, entities, and relations:

```python
for entity in doc.anns.entities:
    ...

for segment in doc.anns.segments:
    ...

for relation in doc.anns.relations:
    ...
```

* get and filter segments, entities and relations:

```python
sentences_segments = doc.get_segments(label="sentences")
disorder_entities = doc.get_entities(label="disorder")

entity = ...
relations = doc.get_relations(label="before", source_id=entity.uid)
```

For more details on common interfaces provided by core components,
please refer to [Document](api:core:document).

### Annotations

For the text modality, {class}`~.text.TextDocument` can only contain multiple {class}`~.text.TextAnnotation`.

Three subclasses are defined {class}`~medkit.core.text.annotation.Segment`,
{class}`~medkit.core.text.annotation.Entity` and {class}`~medkit.core.text.annotation.Relation`.

```{mermaid}
:align: center
:caption: Text annotation hierarchy

classDiagram
     direction TB
    class Annotation{
        <<protocol>>
    }
    class TextAnnotation{
        <<abstract>>
    }
    Annotation <|.. TextAnnotation: implements
    TextAnnotation <|-- Segment
    TextAnnotation <|-- Relation
    Segment <|-- Entity
```

:::{note}
Each text annotation class inherits from the common interfaces
provided by the core component (cf. [Annotation](api:core:annotation)).
:::

For more details about public APIs, please refer to {mod}`medkit.core.text.annotation`.

### Attributes

Text annotations can receive attributes, which will be instances of the core {class}`~.core.Attribute` class.

Among attributes, {mod}`medkit.core.text` proposes {class}`~medkit.core.text.entity_norm_attribute.EntityNormAttribute`,
to be used for normalization attributes, in order to have a common structure for normalization information,
independently of the operation used to create it.

(api:core-text:span)=
## Spans

`medkit` relies on the concept of spans for following all text modifications made by the different operations.

`medkit` also proposes a set of utilities for manipulating these spans when implementing new operations.

For more details about public APIs, please refer to {mod}`medkit.core.text.span`
and {mod}`medkit.core.text.span_utils`.

:::{seealso}
You may also take a look to the [spans examples](../cookbook/spans).
:::

## Text Utilities

These utilities have some preconfigured patterns for preprocessing text documents without destruction.
They are not designed to be used directly, but rather inside a cleaning operation.

For more details about public APIs, please refer to {mod}`medkit.core.text.utils`.

:::{seealso}
`medkit` provides a {class}`~medkit.text.preprocessing.eds_cleaner.EDSCleaner` class,
which combines all these utilities to clean French documents (related to EDS documents coming from PDF).
:::

## Operations

Abstract subclasses of {class}`~.core.Operation` have been defined for text
to ease the development of text operations according to `run` operations.

```{eval-rst}
.. autoclasstree:: medkit.core.operation medkit.core.text.operation
    :strict:
    :namespace: medkit.core
    :align: center
    :caption: Operation hierarchy
```

Internal class `_CustomTextOperation` has been implemented to allow user to
call {func}`~.text.create_text_operation` for easier instantiation of custom
text operations.

For more details about public APIs, please refer to {mod}`medkit.core.text.operation`.

:::{seealso}
Please refer to this [example](../cookbook/custom_text_operation) for examples of custom operation.
:::
