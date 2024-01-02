#  Finding relations, OntoTox implementation

This tutorial shows how to extract chemotherapy toxicities and grading information using the ontotox implementation in medkit.

The [OntoTox](https://github.com/TeamHeka/OntoTox) project proposed a framework to extract information from different sources. In this tutorial, we're going to extract information from free text. The project strongly inspired the development of the {class}`~medkit.text.relations.syntactic_relation_extractor.SyntacticRelationExtractor`  class, a detector of relations between entities using the syntax of the phrase that contains them.

**Implementation**

We divide the implementation in two parts: 
1. Detection of entities: Toxicity and grade
2. Extraction of Relations between entities

:::{note}
The original paper[^footnote1] describes in more detail the implementation. The annotators are operations in medkit. Medkit includes a `QuickUMLSMatcher` but we use `UMLSMatcher`, an easier version to use.
:::

[^footnote1]:Alice Rogier, Adrien Coulet, and Bastien Rance. 2022. Using an Ontological Representation of Chemotherapy Toxicities for Guiding Information Extraction and Integration from EHRs. In Studies in Health Technology and Informatics. IOS Press. DOI:https://doi.org/10.3233/shti220038
___

## Detection of entities

We combine the following operations in a medkit pipeline to get the entities.

- EDSCleaner: Preprocessing module, optional operation to remove extra white spaces. It has some predefined rules, so it may not be useful for all documents.
- SentenceTokenizer: Creates sentences using the clean segments
- ContextDetection: Family, Hypothesis and negation detectors.
- UMLSMatcher: Detects chemotherapy toxicity entities using the ontotox prepared terms
- RegexMatcher: Detects grading information using regular expressions

```python
from medkit.core import DocPipeline, Pipeline, PipelineStep
from medkit.core.text import TextDocument
from medkit.text.context import FamilyDetector, HypothesisDetector, NegationDetector
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule, UMLSMatcher
from medkit.text.preprocessing import EDSCleaner
from medkit.text.relations.syntactic_relation_extractor import SyntacticRelationExtractor
from medkit.text.segmentation import SentenceTokenizer, SyntagmaTokenizer
```

```python

def show_relations(doc):
    relations = doc.anns.get_relations()
    text = "*** Relations information ***\n"
    for rel in relations:
        head = doc.anns.get_by_id(rel.source_id)
        target = doc.anns.get_by_id(rel.target_id)
        text += (
            f"The \"{head.text}\"({head.label}) and"
            f" \"{target.text}\"({target.label}) are related. Dependency: [{rel.metadata['dep_tag']},{rel.metadata['dep_direction']}] \n"
        )
    print(text)


def show_entity_context(entity):
    print(entity.label, ":", entity.text)
    norm_attr = entity.attrs.get(label="NORMALIZATION")
    if norm_attr:
        print("Normalization:", norm_attr[0].value)
    family_attr = entity.attrs.get(label="family")[0]
    print("family:", family_attr.value)
    negation_attr = entity.attrs.get(label="negation")[0]
    print("negation:", negation_attr.value)
    hypothesis_attr = entity.attrs.get(label="hypothesis")[0]
    print("hypothesis:", hypothesis_attr.value)
    print()
```

**Define the DocPipeline**:

We want to add the entities in the document, so we need to use a `DocPipeline` to do it. Let's define this component using the steps for entity detection.

```python
# preprocessing and tokenization
eds_cleaner = EDSCleaner()
sentence_tokenizer = SentenceTokenizer()

# context detection operations (using default FR rules)
family_detector = FamilyDetector(output_label="family")
hypothesis_detector = HypothesisDetector(output_label="hypothesis")

# negation detector: 'negation' should be detected in syntagmes
syntagme_tokenizer = SyntagmaTokenizer(attrs_to_copy=["family", "hypothesis"])
negation_detector = NegationDetector(output_label="negation")

# NER operations
# grading info
regex_rule = RegexpMatcherRule(
    id="id_regexp_grade",
    label="grade",
    regexp=r"\b[Gg][Rr][Aa][Dd][Ee]\s*[0-5]*[I|V]*|\b[Gg]\s*(?:[0-5]|(III|IV|II|I|V))",
    version="2",
    case_sensitive=True,  # must be case sensitive (170522)
)
regex_matcher = RegexpMatcher(
    rules=[regex_rule], attrs_to_copy=["family", "hypothesis", "negation"]
)

# tox info
umls_matcher = UMLSMatcher(
    # Directory containing the UMLS files with all OntoTox terms and concepts
    # You can download the ontotox terms from https://t.ly/a4B-Z
    umls_dir="../data/text/ontotox/umls_data_ontotox",
    cache_dir="../data/text/ontotox/.umls_cache/",
    language="FRE",
    lowercase=True,
    normalize_unicode=True,
    threshold=0.8,
    attrs_to_copy=["family", "hypothesis", "negation"],
)
```

```python
# build the pipeline
pipeline_entities = Pipeline(
    steps=[
        PipelineStep(eds_cleaner, input_keys=["full_text"], output_keys=["clean_text"]),
        PipelineStep(
            sentence_tokenizer, input_keys=["clean_text"], output_keys=["sentences"]
        ),
        PipelineStep(family_detector, input_keys=["sentences"], output_keys=[]),
        PipelineStep(hypothesis_detector, input_keys=["sentences"], output_keys=[]),
        PipelineStep(
            syntagme_tokenizer, input_keys=["sentences"], output_keys=["syntagmes"]
        ),
        PipelineStep(negation_detector, input_keys=["syntagmes"], output_keys=[]),
        PipelineStep(
            umls_matcher, input_keys=["syntagmes"], output_keys=["entities_umls"]
        ),
        PipelineStep(
            regex_matcher, input_keys=["syntagmes"], output_keys=["entities_grade"]
        ),
    ],
    input_keys=["full_text"],
    output_keys=["entities_umls", "entities_grade"],
)
```

**Define `entities_detector` DocPipeline**

As we have already defined the input and output keys in the pipeline, all that remains is to define the `full_text` key : the raw text

```python
entities_detector = DocPipeline(
    pipeline=pipeline_entities,
    labels_by_input_key={"full_text": [TextDocument.RAW_LABEL]},
)
```

```python
# Testing the 'entities_detector' with a phrase
doc = TextDocument("Une dysarthrie de grade 3. Pas de perte de poids")
entities_detector.run([doc])

# Print all entities with their contextual attributes
for entity in doc.anns.entities:
    show_entity_context(entity)
```
```
disorder : dysarthrie
Normalization: umls:C0013362
family: False
negation: False
hypothesis: False

disorder : perte de poids
Normalization: umls:C1262477
family: False
negation: True
hypothesis: False

grade : grade 3
family: False
negation: False
hypothesis: False
```   

## Finding relations

**Define components**

The second part extracts the Relations. In this case, we are looking to find relations between the umls entities with the grade entities. We use the `entities_target` to indicate the target.

:::{note}
You can change the tagger (a spacy nlp object), the `relation_extractor` uses the french tagger by default.
:::

```python
relation_extractor = SyntacticRelationExtractor(
    entities_target=["grade"],relation_label="has_grade"
)
```

**Define the ontotox pipeline**

```python
# build the pipeline
ontoTOX_pipeline = Pipeline(
    steps=[
        PipelineStep(entities_detector, input_keys=["docs"], output_keys=[]),
        PipelineStep(relation_extractor, input_keys=["docs"], output_keys=[]),
    ],
    input_keys=["docs"],
    output_keys=[],
)
```

## Running the OntoTOX pipeline

```python
# Create a doc with the ontotox examples
docs = TextDocument.from_dir("../data/text/ontotox/docs")

# Run the ontotox pipeline
# (1) Find entities and (2) add relations
ontoTOX_pipeline.run(docs)
```

Let's look in detail the first document

```python
doc = docs[0]
print(f"\"{doc.text}\"")

print("*** Entities information *** \n")
for entity in doc.anns.get_entities():
    show_entity_context(entity)
    
# Show relations
show_relations(doc)
```
```
"Pas de perte de poids.
Le patient présente une dysarthrie de grade 3.
Douleurs abdominales    : grade 1.
Pas de toux.
"
*** Entities information *** 

disorder : perte de poids
Normalization: umls:C1262477
family: False
negation: True
hypothesis: False

disorder : dysarthrie
Normalization: umls:C0013362
family: False
negation: False
hypothesis: False

disorder : Douleurs abdominales
Normalization: umls:C0221512
family: False
negation: False
hypothesis: False

disorder : toux
Normalization: umls:C0010200
family: False
negation: True
hypothesis: False

grade : grade 3
family: False
negation: False
hypothesis: False

grade : grade 1
family: False
negation: False
hypothesis: False

*** Relations information ***
The "dysarthrie"(disorder) and "grade 3"(grade) are related. Dependency: [nmod,left_to_right] 
The "Douleurs abdominales"(disorder) and "grade 1"(grade) are related. Dependency: [nsubj,right_to_left] 
```

```python
for doc in docs[1:]:
    print(f"\"{doc.text}\"")
    print("Entities:")
    print("|".join([f"\"{entity.text}\"({entity.label})" 
                   for entity in doc.anns.get_entities()]))
    print()
    show_relations(doc)
    print("\u2500" * 10)
```

```
"Le patient présente une oesophagite peptique de grade 2.
Pas de thrombopénie.
Il présente cependant également une alopécie sévère et un amaigrissment.
Vomissements récurrents.
"
Entities:
"oesophagite"(disorder)|"thrombopénie"(disorder)|"alopécie"(disorder)|"Vomissements"(disorder)|"grade 2"(grade)

*** Relations information ***
The "oesophagite"(disorder) and "grade 2"(grade) are related. Dependency: [nmod,left_to_right] 

──────────
"Le patient présente 


une asthénie de grade 3.
Forte dyspnée.
Une épistaxis 

de grade 4.
"
Entities:
"asthénie"(disorder)|"dyspnée"(disorder)|"épistaxis"(disorder)|"grade 3"(grade)|"grade 4"(grade)

*** Relations information ***
The "asthénie"(disorder) and "grade 3"(grade) are related. Dependency: [nmod,left_to_right] 
The "épistaxis"(disorder) and "grade 4"(grade) are related. Dependency: [nmod,left_to_right] 

──────────
"Thrombose de
    grade 1.
Pas de signe d'anémie.
Elle a fait une 

éruption cutanée.
"
Entities:
"Thrombose"(disorder)|"anémie"(disorder)|"éruption cutanée"(disorder)|"grade 1"(grade)

*** Relations information ***
The "Thrombose"(disorder) and "grade 1"(grade) are related. Dependency: [nmod,left_to_right] 

──────────
"# test2
Pas de perte de poids.
Le patient présente une dysarthrie de grade 3.
Douleurs          abdominales    : Grade 1.
Pas de toux."
Entities:
"perte de poids"(disorder)|"dysarthrie"(disorder)|"Douleurs abdominales"(disorder)|"toux"(disorder)|"grade 3"(grade)|"Grade 1"(grade)

*** Relations information ***
The "dysarthrie"(disorder) and "grade 3"(grade) are related. Dependency: [nmod,left_to_right] 
The "Douleurs abdominales"(disorder) and "Grade 1"(grade) are related. Dependency: [nsubj,right_to_left] 

──────────
```

## Optional: Filter annotations


This is how we have implemented ontotox in medkit, using the extracted information we could now filter out relations containing negated entities, for example. 

:::{note}
Medkit includes some helpers to define simple operations, you may see this [tutorial](../examples/custom_text_operation) for more information.
:::

```python
from medkit.core.text import CustomTextOpType,create_text_operation
```

```python
new_doc = TextDocument("Le patient présente une dysarthrie mais pas de grade 3")
ontoTOX_pipeline.run([new_doc])

# Print all entities with their contextual attributes
for entity in new_doc.anns.entities:
    show_entity_context(entity)
show_relations(new_doc)
print("\u2500" * 10)
```
```
disorder : dysarthrie
Normalization: umls:C0013362
family: False
negation: False
hypothesis: False

grade : grade 3
family: False
negation: True
hypothesis: False

*** Relations information ***
The "dysarthrie"(disorder) and "grade 3"(grade) are related. Dependency: [conj,left_to_right] 

──────────
```
**Define the filter method**

```python
def filter_negated_entities(entity):
    attr = entity.attrs.get(label="negation")
    # only keep entities without negation (negation is false)
    return len(attr)>0 and attr[0].value is False
```

```python
filter_operation = create_text_operation(function=filter_negated_entities, 
                                         function_type=CustomTextOpType.FILTER)
```

**Define the new pipeline**

```python
# add the filter operation
steps_with_filter = entities_detector.pipeline.steps + [PipelineStep(filter_operation,
                                                                    input_keys=["entities_umls","entities_grade"],
                                                                    output_keys=["entities_filtered"],
                                                                    aggregate_input_keys=True)]
# update the pipeline
entities_detector_filter = DocPipeline(
    pipeline=Pipeline(steps=steps_with_filter,
                                     input_keys=["full_text"],
                                     output_keys=["entities_filtered"]),
    labels_by_input_key={"full_text": [TextDocument.RAW_LABEL]},
)

# build the pipeline with the filter
ontoTOX_pipeline_filter = Pipeline(
    steps=[
        PipelineStep(entities_detector_filter, input_keys=["docs"], output_keys=[]),
        PipelineStep(relation_extractor, input_keys=["docs"], output_keys=[]),
    ],
    input_keys=["docs"],
    output_keys=[],
)
```

```python
new_doc = TextDocument("Le patient présente une dysarthrie mais pas de grade 3")
ontoTOX_pipeline_filter.run([new_doc])

# Print all entities with their contextual attributes
for entity in new_doc.anns.entities:
    show_entity_context(entity)
show_relations(new_doc)
print("\u2500" * 10)
```
```
disorder : dysarthrie
Normalization: umls:C0013362
family: False
negation: False
hypothesis: False

*** Relations information ***

──────────
```

:::{seealso}
[Here](../api/text) you can find more information about the text operations.
:::
