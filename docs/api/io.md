# I/O components

This page lists all components for converting and loading/saving data.

:::{note}
For more details about public APIs, refer to
{mod}`medkit.io`.
:::

## medkit-json

medkit has some utilities to export and import medkit documents to json format.

You can use {mod}`medkit.io.medkit_json.save_text_documents` to save a list of documents, and then {mod}`medkit.io.medkit_json.load_text_documents` to load them in medkit.

:::{warning}
`load_text_documents` is a generator function that returns a generator iterator.
That avoids keeping data in memory.

Pay attention that the generator variable becomes empty after the first iteration.

However, if you need to keep all the list in memory, you may cast it to a list:
```python
from medkit.io.medkit_json import load_text_documents

MEDKIT_JSONL_PATH = "path_to_medkit_jsonl_file"

docs = list(load_text_documents(MEDKIT_JSONL_PATH))
```
:::

For more details, refer to {mod}`medkit.io.medkit_json`.

(api:io:brat)=
## Brat

Brat is a web-based tool for text annotation. Medkit supports the **input** and **output** conversion of text documents. 

:::{seealso}
For more details, refer to {mod}`medkit.io.brat`.
You may refer to this [example](../examples/brat_io.md) for more information.
:::


## Doccano

[Doccano](https://github.com/doccano/doccano) is a text annotation tool from multiple tasks. Medkit supports the **input** and **output** conversion of doccano files (.JSONL format). 

You can load annotations from a .jsonl file or a zip directory.

### Supported tasks
| Doccano Project                  	| Task for io converter                                                                                                     	|
|----------------------------------	|---------------------------------------------------------------------------------------------------------------------------	|
| Sequence labeling                	| {class}`medkit.io.doccano.DoccanoTask.SEQUENCE_LABELING` <br> i.e : `{'text':...,'label':[(int,int,label)]}`              	|
| Sequence labeling with relations 	| {class}`medkit.io.doccano.DoccanoTask.RELATION_EXTRACTION` <br>i.e : `{'text':...,'entities':[{...}],'relations':[{...}]}` 	|
| Text Classification              	| {class}`medkit.io.doccano.DoccanoTask.TEXT_CLASSIFICATION`<br>i.e : `{'text':...,'label':[str]}`                          	|

### Client configuration

The doccano user interface allows custom configuration over certain annotation parameters. The {class}`medkit.io.doccano.DoccanoClientConfig` class contains the configuration to be used by the input converter. 

You can modify the settings depending on the configuration of your project. If you don't provide a config, the converter will use the default doccano configuration.


:::{note}
**Metadata**

- Doccano to medkit: All the extra fields are imported as a dictionary in `TextDocument.metadata`
- Medkit to Doccano: The `TextDocument.metadata` is exported as extra fields in the output data. You can set `include_metadata` to False to remove the extra fields.
:::

For more details, refer to {mod}`medkit.io.doccano`.

(api:io:spacy)=
## Spacy

Medkit supports the **input** and **output** conversion of spacy documents.

:::{important}
For using spacy converters, you need to install [spacy](https://spacy.io/).
These dependencies may be installed with `pip install medkit-lib[spacy]`
:::

:::{seealso}
You may refer to this [example](../examples/spacy/index.md) for more information.
:::

For more details, refer to {mod}`medkit.io.spacy`.



## RTTM

Rich Transcription Time Marked (.rttm) files contains diarization information. 
Medkit supports input and output conversion of audio documents.

For more details, refer to {mod}`medkit.io.rttm`.

## SRT

For more details, refer to {mod}`medkit.io.srt`.
