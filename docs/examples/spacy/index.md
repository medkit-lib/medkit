# Spacy integration

[spaCy](https://spacy.io/) is a library for advanced Natural Language Processing in Python. Medkit supports Spacy in input/output conversion as well as annotator. 

| Task                                        | Medkit Operation                                                                        |
| :------------------------------------------ | --------------------------------------------------------------------------------------- |
| Load SpacyDocs                              | {class}`~medkit.io.spacy.SpacyInputConverter`                                           |
| Convert documents to SpacyDocs              | {class}`~medkit.io.spacy.SpacyOutputConverter`                                          |
| Annotate segments using a Spacy pipeline    | {class}`~medkit.text.spacy.pipeline.SpacyPipeline`                                      |
| Annotate documents using a Spacy pipeline   | {class}`~medkit.text.spacy.doc_pipeline.SpacyDocPipeline`                               |
| Detect syntactic relations between entities | {class}`~medkit.text.relations.syntactic_relation_extractor.SyntacticRelationExtractor` |

:::{note}
You may refer to {mod}`medkit.text.spacy` for more information.
:::

```{tableofcontents}
```