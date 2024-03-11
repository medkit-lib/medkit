# Audio Components

This page contains all core audio concepts of `medkit`.

For more details about public APIs, refer to {mod}`medkit.core.audio`.

```{contents} Table of Contents
:depth: 3
```

## Data Structures

The {class}`~medkit.core.audio.document.AudioDocument` class implements the
{class}`~medkit.core.document.Document` protocol.
It allows to store instances of the {class}`~medkit.core.audio.annotation.Segment` class,
which implements the {class}`~medkit.core.annotation.Annotation` protocol.

```{mermaid}
:align: center
:caption: Audio document and annotation hierarchy

classDiagram
    direction TB
    class Document~Annotation~{
        <<protocol>>
    }
    class Annotation{
        <<protocol>>
    }
    class AudioDocument{
        uid: str
        anns: AudioAnnotationContainer
    }
    class Segment {
        uid: str
        label: str
        attrs: AttributeContainer
    }
    Document <|.. AudioDocument: implements
    Annotation <|.. Segment: implements
    AudioDocument *-- Segment : contains\n(AudioAnnotationContainer)

```

### Documents

{class}`~.audio.AudioDocument` relies on {class}`~.audio.AudioAnnotationContainer`,
a subclass of {class}`~medkit.core.AnnotationContainer`, to manage the annotations.

For more details about the common interfaces provided by the core components,
please refer to [Document](api:core:document).

### Annotations

For the audio modality, {class}`~.core.audio.AudioDocument` can only contain {class}`~medkit.core.audio.Segment`.

### Spans

Similarly to [text spans](api:core-text:span), audio annotations have an audio span pointing to the part of
the audio document that is annotated. Contrary to text annotations, multiple discontinuous spans are not supported.

:::{important}
An audio annotation can only have 1 continuous span, and there is no concept of "modified spans".
:::

For more details about public APIs, please refer to {mod}`medkit.core.audio.span`.

## Audio Buffer

Access to the actual waveform data is handled through {class}`~.audio.AudioBuffer` instances.

The same way text annotations store the text they refer to in their `text` property,
audio annotations store the portion of the audio signal they refer to in an `audio` property
holding an {class}`~.audio.AudioBuffer`.

The contents of an {class}`~.audio.AudioBuffer` might be different from the initial raw signal
if it has been preprocessed.
If the signal is identical to the initial raw signal, then a {class}`~.audio.FileAudioBuffer` can be used
(with appropriate `start` and `end` boundaries).
Otherwise, a {class}`~.audio.MemoryAudioBuffer` has to be used, as there is no corresponding audio file containing the signal.

Creating a new {class}`~.audio.AudioBuffer` containing a portion of a pre-existing buffer is done through the `trim()` method.

For more details about public APIs, please refer to {mod}`medkit.core.audio.audio_buffer`.

## Operations

Abstract subclasses of {class}`~.core.Operation` have been defined for audio
to ease the  development of audio operations according to `run` operations.

```{eval-rst}
.. autoclasstree:: medkit.core.operation medkit.core.audio.operation
    :strict:
    :namespace: medkit.core
    :align: center
    :caption: Operation hierarchy
```

For more details about public APIs, please refer to {mod}`medkit.core.audio.operation`.
