# Tools

This page lists miscellaneous utility components.

For more details about public APIs, refer to {mod}`medkit.tools`.

```{contents} Table of Contents
:depth: 3
```

## Provenance

Helper function to generate a [Graphviz](https://graphviz.org/docs/layouts/dot/) layout from provenance data
and save it to the dot format for easier visualization.

Refer to {func}`medkit.tools.save_prov_to_dot`.

## HuggingFace

Helper functions for operations using [HuggingFace](https://huggingface.co/) models.

Refer to {mod}`medkit.tools.hf_utils`.

## mtsamples

Helper functions to facilitate downloads of `mtsamples` data into a cache directory
before loading and converting them to `medkit` format.

For example, to load the first ten sample text documents:

```python
from medkit.tools.mtsamples import load_mtsamples

docs = load_mtsamples(nb_max=10)
```

For more details about `mtsamples` data, please refer to {mod}`medkit.tools.mtsamples`.

## E3C corpus

The E3C corpus may be downloaded from:
- the [E3C project's website](https://live.european-language-grid.eu/catalogue/corpus/7618/download/);
- the [E3C GitHub releases](https://github.com/hltfbk/E3C-Corpus/releases).

Once downloaded and unzipped, you may:

- load the data collection into `medkit` text documents:

```python
from pathlib import Path
from medkit.tools.e3c_corpus import load_data_collection

data_collection_layer1 = Path("/tmp/E3C-Corpus-2.0.0/data_collection/French/layer1")

docs = load_data_collection(data_collection_layer1)
```

- convert the data collection to `medkit` text documents:

```python
from pathlib import Path
from medkit.tools.e3c_corpus import convert_data_collection_to_medkit

data_collection = Path("/tmp/E3C-Corpus-2.0.0/data_collection/French")
layers = ["layer1", "layer2", "layer3"]

for layer in layers:
    dir_path = data_collection / layer
    medkit_file = f"medkit_e3c_{layer}.jsonl"
    convert_data_collection_to_medkit(
        dir_path=dir_path, output_file=medkit_file
    )
```

- load the annotated data into `medkit` text documents:

```python
from pathlib import Path
from medkit.tools.e3c_corpus import load_data_annotation

data_annotation_layer1 = Path("/tmp/E3C-Corpus-2.0.0/data_annotation/French/layer1")

docs = load_data_annotation(data_annotation_layer1)
```

- convert the annotated data to `medkit` text documents.

```python
from pathlib import Path
from medkit.tools.e3c_corpus import convert_data_annotation_to_medkit

data_annotation = Path("/tmp/E3C-Corpus-2.0.0/data_annotation/French")
layers = ["layer1", "layer2"]

for layer in layers:
    dir_path = data_annotation / layer
    medkit_file = f"medkit_e3c_annotated_{layer}.jsonl"
    convert_data_annotation_to_medkit(
        dir_path=dir_path, output_file=medkit_file
    )
```

For more details about E3C corpus data, please refer to {mod}`medkit.tools.e3c_corpus`.
