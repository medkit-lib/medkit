# medkit

`medkit` is a Python library which facilitates **extraction of features**
from various modalities of patient data, including text and audio for now
-- relational, image, genetic, and others will follow soon.

To this end, medkit enables composition of pipelines with multiple modules,
developed by us, yourself or others.

`medkit` places a strong emphasis on **non-destructive operations**,
i.e. no loss of information when passing data from a module to another,
and a flexible tracing of **data provenance**.

`medkit` aims at accelerating the development of a learning health system,
with a strong dedication to open-source and community development.

:::{warning}
The `medkit` core library is still under heavy development and testing.
Some public interfaces may change in the future.
Please check the **BREAKING CHANGES** section of the project's changelog for details.
:::

:::{toctree}
---
caption: User Guide
hidden:
titlesonly:
---
user_guide/install
user_guide/first_steps
user_guide/pipeline
user_guide/provenance
user_guide/module
:::

:::{toctree}
---
caption: Examples
hidden:
titlesonly:
---
examples/spans
examples/cleaning_text
examples/brat_io
examples/spacy/index
examples/custom_text_operation
examples/edsnlp
examples/iamsystem
examples/finetuning_hf_model
examples/detecting_text_duplicates
examples/audio_transcription
examples/audio_dataset_metrics
examples/ontotox
:::

:::{toctree}
---
caption: Tutorial
hidden:
titlesonly:
---
tutorial/context_detection
tutorial/entity_matching
tutorial/text_segmentation/index
:::

:::{toctree}
---
caption: Reference
hidden:
titlesonly:
---
api/_generated/index
api/audio
api/core
api/core_audio
api/core_text
api/io
api/text
api/training
api/tools
changelog
:::
