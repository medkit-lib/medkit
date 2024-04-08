# Benchmark of NER methods on French clinical corpora

Comparison of NER approaches on French clinical text,
submitted as a full-paper for the MIE 2024 conference.

This comparison makes use of the following pipelines:

| Pipeline                         | Description                                 |
|----------------------------------|---------------------------------------------|
| [Preprocessing](./preprocessing) | Convert, filter and tokenize source corpora |
| [Training](./training)           | Fine-tuning of BERT-based matchers          |
| [Evaluation](./evaluation)       | Evaluate and compare NER methods            |

:::{toctree}
:hidden:
preprocessing
training
evaluation
:::
