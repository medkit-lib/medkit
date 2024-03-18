# I/O Components

This page lists all components for converting and loading/saving data.

For more details about public APIs, please refer to {mod}`medkit.io`.

```{contents} Table of Contents
:depth: 3
```

## medkit-json

`medkit` has some utilities to export and import documents saved as JSON format.

You can use {mod}`medkit.io.medkit_json.save_text_documents` to save a list of documents,
and then {mod}`medkit.io.medkit_json.load_text_documents` to load them within `medkit`.

:::{warning}
`load_text_documents` is a generator function yielding a single document per iteration,
to prevent accidental memory spikes if the corpus is large.

To load the full corpus in memory, you may consume the generator in a list with:

```python
from medkit.io.medkit_json import load_text_documents

docs = list(load_text_documents("/path/to/medkit/documents.jsonl"))
```
:::

For more details, please refer to {mod}`medkit.io.medkit_json`.

(api:io:brat)=
## Brat

[Brat](https://brat.nlplab.org) is a web-based tool for text annotation.
`medkit` supports input and output conversions of Brat text documents. 

For more details about the public API, please refer to {mod}`medkit.io.brat`.

:::{seealso}
You may refer to this [example](../cookbook/brat_io.md) for more information.
:::

## Doccano

[Doccano](https://github.com/doccano/doccano) is a text annotation tool from multiple tasks.
`medkit` supports input and output conversions of doccano files (saved in JSONL format). 

You can load annotations from a JSONL file or a ZIP directory.

### Supported tasks

```{list-table}
* - Doccano Project
  - Task for converter
  - Example
* - Sequence labeling
  - {class}`medkit.io.doccano.DoccanoTask.SEQUENCE_LABELING`
  - `{'label': [(int, int, str)], 'text': ...}`{l=python}
* - Relation extraction
  - {class}`medkit.io.doccano.DoccanoTask.TEXT_CLASSIFICATION`
  - `{'label': [str], 'text': ...}`{l=python}
```

### Client Configuration

The doccano user interface allows custom configuration over certain annotation parameters.
The {class}`medkit.io.doccano.DoccanoClientConfig` class contains the configuration to be used by the input converter. 

You can modify the settings depending on the configuration of your project.
If no custom configuration is provided, the converter will use the default doccano configuration.

:::{note} Metadata

- Doccano to `medkit`: All the extra fields are imported as a dictionary in `TextDocument.metadata`
- `medkit` to Doccano: The `TextDocument.metadata` are exported as extra fields to the output data.
  Set `include_metadata` to `False` to exclude the extra fields.
:::

For more details, please refer to {mod}`medkit.io.doccano`.

(api:io:spacy)=
## spaCy

`medkit` supports input and output conversions of spaCy documents.

:::{important}
Using spaCy converters requires additional dependencies:

```console
pip install 'medkit-lib[spacy]'
```
:::

:::{seealso}
You may refer to this [example](../cookbook/spacy/index.md) for more information.
:::

For more details, please refer to {mod}`medkit.io.spacy`.

## RTTM

Rich Transcription Time Marked files (saved with .rttm extension) contains diarization information. 
`medkit` supports input and output conversions of audio documents in RTTM format.

For more details, refer to {mod}`medkit.io.rttm`.

## SRT

SRT files (saved with .srt extension) contains transcription information associated with an audio recording.
`medkit` supports input and output conversions of audio transcription in SRT format.

:::{important}
Using SRT converters requires additional dependencies:

```console
pip install 'medkit-lib[srt-io-converter]'
```
:::

For more details, refer to {mod}`medkit.io.srt`.
