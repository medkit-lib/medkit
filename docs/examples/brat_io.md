---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Brat integration

+++

Brat is a web-based tool for text annotation that uses the [standoff format](https://brat.nlplab.org/standoff.html). Medkit supports input-output conversion of Brat files with the following annotations types:

- Entities 
- Relations
- Attributes

Annotations with other types are ignored in the conversion process.

In this example, we will show how to import Brat annotated files into medkit and how to convert medkit documents into Brat annotated collections.

Consider this text file: 

+++
```{code-cell} ipython3
# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/examples/input/brat/doc_01.txt

from pathlib import Path

print(Path("./input/brat/doc_01.txt").read_text(encoding="utf-8"))
```

+++

It has the following brat annotation file:

```{code-cell} ipython3
# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/examples/input/brat/doc_01.ann

print(Path("./input/brat/doc_01.ann").read_text(encoding="utf-8"))
```

## Load brat into a list of TextDocuments

To load Brat Files, medkit provides the {class}`~medkit.io.brat.BratInputConverter` class. This converter returns a list of `TextDocument`. 

```{tip}
You can enable provenance tracing by assigning a {class}`~medkit.core.ProvTracer` object to the BratInputConverter with the `set_prov_tracer()` method.
```

```{code-cell} ipython3
from medkit.io.brat import BratInputConverter

# Define Input Converter 
brat_converter = BratInputConverter()

# Load brat into a list of documents
docs = brat_converter.load(dir_path="./input/brat")
medkit_doc = docs[0]

# Explore annotations
print(f"The document has {len(medkit_doc.anns)} annotations")
entities_disease = medkit_doc.anns.get(label="disease")
print(f"Where {len(entities_disease)} annotations have 'disease' as label")

```

**Visualize entities information**

The created document contains the annotations defined in the brat annotation file. 
We can show the entities information, for example.

```{code-cell} ipython3
for entity in medkit_doc.anns.get_entities():
    print(f"label={entity.label}, spans={entity.spans}, text={entity.text!r}")
```

## Save TextDocuments to Brat

To save a list of `TextDocument` in Brat format, you can use {class}`~medkit.io.brat.BratOutputConverter`.

You can choose which medkit **annotations** and **attributes** to keep in the resulting Brat collection. By default, since its `anns_labels` and `attrs` are set to `None`, all annotations and attributes will be in the generated file. 

If you also want to include the segments in the brat collection, the parameter `ignore_segments` can be set to `False`.
+++

**Automatic configuration of annotations**
+++
> Brat is actually controlling the configuration with text-based configuration files. It uses four types, but only the annotation types configuration is necessary (cf: [brat configuration](https://brat.nlplab.org/configuration.html)).

To facilitate integration and ensure correct visualisation, medkit automatically generates an `annotation.conf` for each collection.
 
+++
```{code-cell} ipython3
from medkit.io.brat import BratOutputConverter

# Define Output Converter with default params,
# transfer all annotations and attributes
brat_output_converter = BratOutputConverter()

# save the medkit documents in `dir_path`
brat_output_converter.save(
  docs,  dir_path="./brat_out", doc_names=["doc_1"])
```

The collection is saved on disk including the following files:
* `doc_1.txt`: text of medkit document
* `doc_1.ann`: brat annotation file
* `annotation.conf`: annotation type configuration

By default the name is the `document_id`, you can change it using the `doc_names` parameter.

```{note}
Since the values of the attributes in brat must be defined in the configuration, medkit shows the top50 for each attribute. In case you want to show more values in the configuration, you can change `top_values_by_attr` in the brat output converter.
 ```

:::{seealso}
cf. [Brat IO module](api:io:brat).
:::
