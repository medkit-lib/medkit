# Spacy integration

[spaCy](https://spacy.io/) is a library for advanced Natural Language Processing in Python.

`medkit` supports `spaCy` through input and output conversions as well as annotators. 

| Task                                        | Operation                                                                               |
|:--------------------------------------------|-----------------------------------------------------------------------------------------|
| Load a spaCy Doc                            | {class}`~medkit.io.spacy.SpacyInputConverter`                                           |
| Convert documents to spaCy Doc              | {class}`~medkit.io.spacy.SpacyOutputConverter`                                          |
| Annotate segments using a spaCy pipeline    | {class}`~medkit.text.spacy.pipeline.SpacyPipeline`                                      |
| Annotate documents using a spaCy pipeline   | {class}`~medkit.text.spacy.doc_pipeline.SpacyDocPipeline`                               |
| Detect syntactic relations between entities | {class}`~medkit.text.relations.syntactic_relation_extractor.SyntacticRelationExtractor` |

:::{note}
You may refer to {mod}`medkit.text.spacy` for more information.
:::

:::{toctree}
spacy_io
spacy_pipeline
:::
