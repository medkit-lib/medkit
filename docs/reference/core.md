# Core Components

This page explains all core concepts defined in `medkit`.

For more details, please refer to {mod}`medkit.core`.

```{contents} Table of Contents
:depth: 3
```

## Data Structures

`medkit` document classes are used to access raw data,
as well as store annotations extracted from these raw data.

The {class}`~.core.Document` and {class}`~.core.annotation.Annotation` protocols
are defined inside {mod}`medkit.core`.
They define common properties and methods across all modalities.
These protocols are then implemented for each modality (text, audio, image, etc.),
with additional logic specific to the modality.

To facilitate the implementation of the {class}`~.core.Document` protocol,
the {class}`~.core.AnnotationContainer` class is provided.
It behaves like a list of annotations, with additional filtering methods
and support for non-memory storage.

{mod}`medkit.core` also defines the {class}`~.core.Attribute` class,
which can be used to attach attributes to annotations for any modality.
Similarly to {class}`~.core.AnnotationContainer`, the role of this container is
to provide additional methods for facilitating access to the list of attributes
belonging to an annotation.

```{mermaid}
:align: center
:caption: Core protocols and classes

classDiagram
    direction LR
    class Document~Annotation~{
        <<protocol>>
        uid: str
        anns: AnnotationContainer~Annotation~
    }
    class Annotation{
        <<protocol>>
        uid: str
        label: str
        attrs: AttributeContainer
    }
    class Attribute{
        uid: str
        label: str
        value: Optional[Any]
    }
    Document *-- Annotation : contains\n(AnnotationContainer)
    Annotation *-- Attribute : contains\n(AttributeContainer)
```

Currently, {mod}`medkit.core.text` implements a {class}`~.text.TextDocument` class
and a corresponding set of {class}`~.text.TextAnnotation` subclasses.
Similarly, {mod}`medkit.core.audio` provides an {class}`~.audio.AudioDocument` class
and a corresponding {class}`~medkit.core.audio.annotation.Segment`.
Both modality are also subclasses of {class}`~.core.AnnotationContainer`
to provide some modality-specific logic or filtering.

You may refer to the documentation specific to [audio](core_audio.md) and [text](core_text.md) modalities.

(api:core:document)=
### Document

The {class}`~.core.Document` protocol defines the minimal data structure for a `medkit` document.
Regardless of the modality, each document is linked to a corresponding annotation container.

The {class}`~.core.AnnotationContainer` class provides a set of methods to be implemented for each modality.

The goal is to provide users with a minimum set of common interfaces
for accessing to the document annotations whatever the modality.

Given a document named `doc`, one can:

- browse its annotations:

```python
for ann in doc.anns:
    ...
```

- add a new annotation:

```python
doc.anns.add(...)
```

- get annotations filtered by label:

```python
disorders = doc.anns.get(label="disorder")
```

For more details about the public API, please refer to {class}`medkit.core.document.Document`
and {class}`medkit.core.annotation_container.AnnotationContainer`.

(api:core:annotation)=
### Annotations and Attributes

The {class}`~medkit.core.annotation.Annotation` protocol class provides the minimal data structure
for a `medkit` annotation. Each annotation is linked to an attribute container.

The {class}`~.core.AttributeContainer` class provides a set of common interfaces
for accessing attributes (`~.core.Attribute`) associated to an annotation,
regardless of the underlying modality.

Given an annotation `ann`, one can:

- browse the annotation attributes:

```python
for attr in ann.attrs:
    ...
```

- add a new attribute

```python
ann.attrs.add(...)
```

- get attributes filtered by label:

```python
normalized = ann.attrs.get(label="NORMALIZATION")
```

(api:core:operations)=
## Operations

The {class}`~.core.Operation` abstract class groups all necessary methods for
being compatible with `medkit` processing pipeline and provenance.

We have defined different subclasses depending on the nature of the operation,
including text-specific and audio-specific operations in {mod}`medkit.core.text`
and {mod}`medkit.core.audio`.

To get more details about each modality, you can refer to their documentation:
* [core text](core_text.md)
* [core audio](core_audio.md)

For all operations inheriting from {class}`~.core.Operation` abstract class,
these 4 lines shall be added in `__init__` method:

```python
def __init__(self, ..., uid=None):
    ...
    # Pass all arguments to super (remove self)
    init_args = locals()
    init_args.pop("self")
    super().__init__(**init_args)
```

Each operation is described with {class}`~.core.OperationDescription`.

## Converters

Two abstract classes have been defined for managing document conversion
between `medkit` format and another one.

For more details about the public APIs, refer to {mod}`medkit.core.conversion`.

(api:core:pipeline)=
## Pipeline

{class}`~.core.Pipeline` allows to chain several operations.

To better understand how to declare and use `medkit` pipelines, you may refer
to the [pipeline tutorial](../user_guide/pipeline).

The {class}`~medkit.core.doc_pipeline.DocPipeline` class is a wrapper allowing
to run an annotation pipeline on a list of documents by automatically attach
output annotations to these documents.

For more details about the public APIs, refer to {mod}`medkit.core.pipeline`.

(api:core:provenance)=
## Provenance

:::{warning}
This work is still under development. It may be changed in the future.
:::

Provenance is a `medkit` concept allowing to track all operations and
their role in new knowledge extraction.

With this mechanism, we will be able to provide the provenance information
about a generated data. To log this information, a separate provenance
store is used.

For better understanding this concept, you may follow the
[provenance tutorial](../user_guide/provenance) and/or refer to
["how to make your own module"](../user_guide/module) to know what you have to
do to enable provenance.

For more details about the public APIs, refer to {mod}`medkit.core.prov_tracer`.
