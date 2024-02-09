# Overview

`medkit` is a Python library which facilitates **extraction of features**
from various modalities of patient data, including text and audio for now
-- relational, image, genetic, and others will follow soon.

`medkit` places a strong emphasis on **non-destructive operations**,
i.e. no loss of information when passing data from a module to another,
and a flexible tracing of **data provenance**.
It enables composition of pipelines with multiple modules,
developed by the _HeKA Research Team_, contributors, and eventually yourself.

`medkit` aims at accelerating the development of a learning health system,
with a strong dedication to open-source and community development.

::::{grid} 2
:gutter: 2

:::{grid-item-card} {octicon}`people;2em;sd-mr-1` User Guide

To get started with `medkit`

+++
[Learn more »](user_guide/first_steps)
:::

:::{grid-item-card} {octicon}`rocket;2em;sd-mr-1` Tutorial

To walk through `medkit` features

+++
[Learn more »](tutorial/entity_matching)
:::

:::{grid-item-card} {octicon}`book;2em;sd-mr-1` Cookbook

To learn `medkit` by examples

+++
[Learn more »](cookbook/spans)
:::

:::{grid-item-card} {octicon}`search;2em;sd-mr-1` Reference

For developers and contributors

+++
[Learn more »](reference/core)
:::
::::

:::{warning}
The `medkit` core library is still under heavy development and testing.
Some public interfaces may change in the future.
Please check the **BREAKING CHANGES** section of the project's changelog for details.
:::

```{toctree}
:caption: 👥 User Guide
:hidden:
:titlesonly:

user_guide/install
user_guide/first_steps
user_guide/pipeline
user_guide/provenance
user_guide/module
```

```{toctree}
:caption: 🚀 Tutorial
:hidden:
:titlesonly:

tutorial/context_detection
tutorial/entity_matching
tutorial/text_segmentation/index
```

```{toctree}
:caption: 📖 Cookbook
:hidden:
:titlesonly:

cookbook/spans
cookbook/cleaning_text
cookbook/brat_io
cookbook/spacy/index
cookbook/custom_text_operation
cookbook/edsnlp
cookbook/iamsystem
cookbook/finetuning_hf_model
cookbook/detecting_text_duplicates
cookbook/audio_transcription
cookbook/audio_dataset_metrics
cookbook/ontotox
cookbook/ner_benchmark/index
```

```{toctree}
:caption: 🔍 Reference
:hidden:
:titlesonly:

reference/api/index
reference/audio
reference/core
reference/core_audio
reference/core_text
reference/io
reference/text
reference/tools
reference/training
changelog
license
```
