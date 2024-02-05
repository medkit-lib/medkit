# Text operation modules

This page lists all components related to text processing.

:::{note}
For more details about all sub-packages, refer to
{mod}`medkit.text`.
:::

## Summary

Here is a list of all the medkit text operations with a direct link to the corresponding API docs. Scrolling down this page, you will find more details about the components of `medkit.text`.

**Preprocessing:**
:::{list-table}
*   - {mod}`CharReplacer<medkit.text.preprocessing.char_replacer>`
    - Fast replacement of 1-char string by n-char strings
*   - {mod}`RegexpReplacer<medkit.text.preprocessing.regexp_replacer>`
    - Patterns replacement
*   - {mod}`EDSCleaner<medkit.text.preprocessing.eds_cleaner>`
    - Cleaning of texts extracted from the APHP EDS
*   - {mod}`DuplicateFinder<medkit.text.preprocessing.duplicate_finder>`
    - Detection of duplicated parts across documents based on [duptextfinder](https://github.com/equipe22/duplicatedZoneInClinicalText/)
:::

**Segmentation:**
:::{list-table}
*   - {mod}`SectionTokenizer<medkit.text.segmentation.section_tokenizer>`
    - Rule-based detection of sections
*   - {mod}`SentenceTokenizer<medkit.text.segmentation.sentence_tokenizer>`
    - Rule-based sentence splitting
*   - {mod}`RushSentenceTokenizer<medkit.text.segmentation.rush_sentence_tokenizer>`
    - Sentence splitting based on [PyRuSH](https://github.com/jianlins/PyRuSH/)
*   - {mod}`SyntagmaTokenizer<medkit.text.segmentation.syntagma_tokenizer>`
    - Rule-based sub-sentence splitting
:::

**Context:**
:::{list-table}
*   - {mod}`NegationDetector<medkit.text.context.negation_detector>`
    - Detection of negation
*   - {mod}`HypothesisDetector<medkit.text.context.hypothesis_detector>`
    - Detection of hypothesis
*   - {mod}`FamilyDetector<medkit.text.context.family_detector>`
    - Detection of family antecedents
:::

**Named Entity Recognition:**
:::{list-table}
*   - {mod}`RegexpMatcher<medkit.text.ner.regexp_matcher>`
    - Regexp-based entity matching
*   - {mod}`SimstringMatcher<medkit.text.ner.simstring_matcher>`
    - Fast fuzzy matching based on [simstring](http://chokkan.org/software/simstring/)
*   - {mod}`IAMSystemMatcher<medkit.text.ner.iamsystem_matcher>`
    - Advanced entity matching based on [IAMSystem](https://github.com/scossin/iamsystem_python)
*   - {mod}`UMLSMatcher<medkit.text.ner.umls_matcher>`
    - Matching of [UMLS](https://www.nlm.nih.gov/research/umls) terms based on [simstring](http://chokkan.org/software/simstring/)
*   - {mod}`QuickUMLSMatcher<medkit.text.ner.quick_umls_matcher>`
    - Matching of [UMLS](https://www.nlm.nih.gov/research/umls/) terms based on [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS)
*   - {mod}`HFEntityMatcher<medkit.text.ner.hf_entity_matcher>`
    - Entity matcher relying on [HuggingFace transformers](https://huggingface.co/docs/transformers/) models
*   - {mod}`DucklingMatcher<medkit.text.ner.duckling_matcher>`
    - General matcher (dates, quantities, etc) relying on [Duckling](https://github.com/facebook/duckling)
coder normalizer
*   - {mod}`EDSNLPDateMatcher<medkit.text.ner.edsnlp_date_matcher>`
    - Date/time matching based on [EDS-NLP](https://aphp.github.io/edsnlp/)
*   - {mod}`EDSNLPTNMMatcher<medkit.text.ner.edsnlp_tnm_matcher>`
    - TNM (Tumour/Node/Metastasis) matching based on [EDS-NLP](https://aphp.github.io/edsnlp/)
*   - {mod}`UMLSCoderNormalizer<medkit.text.ner.umls_coder_normalizer>`
    - Normalization of pre-existing entities to [UMLS](https://www.nlm.nih.gov/research/umls/) CUIs relying on a [CODER model](https://github.com/GanjinZero/CODER)
*   - {mod}`NLStructEntityMatcher<medkit.text.ner.nlstruct_entity_matcher>`
    - Entity matcher relying on [NLStruct](https://github.com/percevalw/nlstruct) models.
:::

**spaCy:**
:::{list-table}
*   - {mod}`SpacyPipeline<medkit.text.spacy.pipeline>`
    - Operation wrapping a [spaCy](https://spacy.io/) pipeline to work at the annotation level
*   - {mod}`SpacyDocPipeline<medkit.text.spacy.doc_pipeline>`
    - Operation wrapping a [spaCy](https://spacy.io/) pipeline to work at the document level
*   - {class}`~medkit.text.spacy.edsnlp.EDSNLPPipeline`
    - Operation wrapping an [EDS-NLP](https://aphp.github.io/edsnlp/) pipeline to work at the annotation level
*   - {class}`~medkit.text.spacy.edsnlp.EDSNLPDocPipeline`
    - Operation wrapping an [EDS-NLP](https://aphp.github.io/edsnlp/) pipeline to work at the document level
:::

**Misc:**
:::{list-table}
*   - {mod}`SyntacticRelationExtractor<medkit.text.relations.syntactic_relation_extractor>`
    - Relation detector relying on [spaCy](https://spacy.io/)'s dependency parser
*   - {mod}`HFTranslator<medkit.text.translation.hf_translator>`
    - Translation operation relying on [HuggingFace transformers](https://huggingface.co/docs/transformers/) models
*   - {mod}`AttributeDuplicator<medkit.text.postprocessing.attribute_duplicator>`
    - Propagation of attributes based on annotation spans
*   - {mod}`DocumentSplitter<medkit.text.postprocessing.document_splitter>`
    - A component to divide text documents using its segments as a reference
:::

## Pre-processing modules

This section provides some information about how to use preprocessing modules.

:::{note}
For more details about public API, refer to
{mod}`medkit.text.preprocessing`.
:::

If you need to pre-process your document texts for replacing some sub-texts by
other ones, medkit provides some operations to do that and keep span information.

If you want to use some rule-based operations (like
{class}`~.text.ner.RegexpMatcher` for example), document texts may need to be
pre-processed.

For example, concerning the {class}`~.text.ner.RegexpMatcher`:

> When the rule is not sensitive to unicode, we try to convert unicode chars
> to the closest ascii chars. However, some characters need to be pre-processed
> before (e.g., `n°` -> `number`). So, if the text lengths are different, we
> fall back on initial unicode text for detection even if rule is not
> unicode-sensitive.
> In this case, a warning is logged for recommending to pre-process data.

### CharReplacer

{class}`~.text.preprocessing.CharReplacer` is a pre-processing operation allowing
to replace one character by another one.
It is faster than {class}`~.text.preprocessing.RegexpReplacer` but is limited to
character replacement and does not support pattern replacement.

For example, if you want to replace some special characters like `+`:

```
from medkit.core.text import TextDocument
from medkit.text.preprocessing import CharReplacer

doc = TextDocument(text="Il suit + ou - son traitement,")

rules = [("+", "plus"), ("-", "moins")]
op = CharReplacer(output_label="preprocessed_text", rules=rules)
new_segment = op.run([doc.raw_segment])[0]
print(new_segment.text)
```

Results:
* `new_segment.text` : "Il suit plus ou moins son traitement,"
* `new_segment.spans` : [Span(start=0, end=8),
            ModifiedSpan(length=4, replaced_spans=[Span(start=8, end=9)]),
            Span(start=9, end=13),
            ModifiedSpan(length=5, replaced_spans=[Span(start=13, end=14)]),
            Span(start=14, end=30)]

medkit also provides some pre-defined rules that you can import
(cf. {mod}`medkit.text.preprocessing.rules`) and then combine with your own rules.

For example:
```
from medkit.text.preprocessing import (
    CharReplacer,
    LIGATURE_RULES,
    SIGN_RULES,
    SPACE_RULES,
    DOT_RULES,
    FRACTION_RULES,
    QUOTATION_RULES,
)

rules = (
    LIGATURE_RULES
    + SIGN_RULES
    + SPACE_RULES
    + DOT_RULES
    + FRACTION_RULES
    + QUOTATION_RULES
    + <my_own_rules>
)

# same as rules = ALL_RULES + <my_own_rules>

op = CharReplacer(output_label="preprocessed_text", rules=rules)
```

:::{note}
If you do not provide rules when initializing char replacer operation, 
all pre-defined rules (i.e., ALL_RULES) are used. 
:::


### RegexpReplacer

The {class}`~.text.preprocessing.RegexpReplacer` operation uses an algorithm based
on regular expressions for detecting patterns in the text and replace them by
the new text, and all that with preserving span information.
So, it may be useful if you need to replace sub-text or text with a context by
other ones.

For example, if you want to replace `n°` by `numéro`:

```
from medkit.core.text import TextDocument
from medkit.text.preprocessing import RegexpReplacer

doc = TextDocument(text="À l'aide d'une canule n ° 3,")

rule = (r"n\s*°", "numéro")
op = RegexpReplacer(output_label="preprocessed_text", rules=[rule])
new_segment = op.run([doc.raw_segment])[0]
print(new_segment.text)
```

Results:
* `new_segment.text` : "À l'aide d'une canule numéro 3,"
* `new_segment.spans` : [Span(start=0, end=22),
 ModifiedSpan(length=6, replaced_spans=[Span(start=22, end=25)]),
 Span(start=25, end=28)]

:::{warning}
If you have a lot of single characters to change, it is not the optimal way to
do it for performance reasons.
In this case, we recommend to use {class}`~.text.preprocessing.CharReplacer`.
:::

### Other pre-processing modules

medkit also provides an operation for cleaning up text. This module has been
implemented for a specific case of EDS document.

You can follow this [tutorial example](../examples/cleaning_text.md) for more
details about this {class}`~.text.preprocessing.EDSCleaner` module.

(api:text:segmentation_modules)=
## Segmentation modules

This section lists text segmentation modules. They are part of
{mod}`medkit.text.segmentation` package.

:::{note}
For more details about public APIs of each module, refer to
{mod}`medkit.text.segmentation` sub-modules.
:::

{class}`~.text.segmentation.SectionTokenizer` and
{class}`~.text.segmentation.SyntagmaTokenizer` may rely on a description file
containing the set of user-defined rules for splitting document text into a list
of medkit {class}`~.text.Segment` corresponding successively to sections or
syntagmas.

For {class}`~.text.segmentation.SectionTokenizer`, here is the yaml schema 
reference of the file.

* `sections` : dictionary of key-values where *key* will be the section name and
*value* is a list of keywords to detect as the start of the section.
* `rules` : list of modification rules which role is to rename a detected section
  * `rules.section_name` : name of the detected section to rename
  * `rules.new_section_name` : new name wanted for the section
  * `rules.order`: order condition for renaming. Possible values: BEFORE, AFTER
  * `other_sections` : list of other section names (i.e., context of the section
  to rename) to use with the order condition

:::{note}
You may test French default rules using `section_tokenizer = SectionTokenizer()`.
The corresponding file content is available
[here](https://raw.githubusercontent.com/medkit-lib/medkit/develop/medkit/text/segmentation/default_section_definition.yml).
:::

For {class}`~.text.segmentation.SyntagmaTokenizer`, here is the yaml schema 
reference of the file.

* `syntagma.separators` : list of regular expressions allowing to trigger the
start of a new syntagma.

:::{note}
You may test default French rules using `syntagma_tokenizer = SyntagmaTokenizer()`.
The corresponding file content is available
[here](https://raw.githubusercontent.com/medkit-lib/medkit/develop/medkit/text/segmentation/default_syntagma_definition.yml).
:::


```{admonition} Examples
For a better understanding, you may follow these tutorial examples:
* section: [section tokenizer tutorial](../examples/text_segmentation/section)
* syntagma: [syntagma tokenizer tutorial](../examples/text_segmentation/syntagma)
* sentence: [first steps tutorial](../user_guide/first_steps)
```


## Context detection modules

This section lists text annotators for detecting context. They are part of
{mod}`medkit.text.context` package.

### Hypothesis

If you want to test default French rules of {class}`~.text.context.HypothesisDetector`,
you may use :
```
detector = HypothesisDetector()
detector.run(syntagmas)
```

:::{note}
For more details about public APIs, refer to
{mod}`~.text.context.hypothesis_detector`.
:::

### Negation

medkit provides a rule-based negation detector which attaches a negation
attribute to a text segment.

:::{note}
For more details about public APIs, refer to
{mod}`~.text.context.negation_detector`.
:::

### Family reference

medkit provides a rule-based family detector which attaches a family
attribute to a text segment.

:::{note}
For more details about public APIs, refer to
{mod}`~.text.context.family_detector`.
:::

## NER modules

This section lists text annotators for detecting entities. They are part of
{mod}`medkit.text.ner` package.


### Regular Expression Matcher

medkit provides a rule-based entity matcher.

For an example of {class}`~.text.ner.RegexpMatcher` usage, you can follow this
[example tutorial](../user_guide/first_steps.md).

:::{note}
For more details about public APIs, refer to {mod}`~.text.ner.regexp_matcher`.
:::

### IAM system Matcher

The [iamsystem library](https://iamsystem-python.readthedocs.io/en/latest/) is
available under the following medkit operation.

:::{note}
For more details about public APIs, refer to {mod}`~.text.ner.iamsystem_matcher`.
:::

---

medkit also provides a custom implementation ({class}`~.text.ner.MedkitKeyword`) of
[IAM system IEntity](https://iamsystem-python.readthedocs.io/en/latest/getstarted.html#with-a-custom-of-keyword-subclass)
which allows user:
* to associate `kb_name` to `kb_id`
* to provide a medkit entity label (e.g., category) associated to the IAM system entity label (i.e., text to search).

```{admonition} Examples
For a better understanding, you may follow the
 [iamsystem matcher example tutorial](../examples/iamsystem)
```

### Simstring Matcher

Medkit provides an entity matcher using the [simstring](http://chokkan.org/software/simstring/) fuzzy matching algorithm.

:::{note}
For more details about public APIs, refer to {mod}`~.text.ner.simstring_matcher`.
:::


### Quick UMLS Matcher

:::{important}
{class}`~.quick_umls_matcher.QuickUMLSMatcher` needs additional dependencies that can be installed with `pip
install medkit[quick-umls-matcher]`


QuickUMLSMatcher is a wrapper around 3d-party quickumls.core.QuickUMLS,
which requires a QuickUMLS install to work. A QuickUMLS install can be
created with
```
python -m quickumls.install <umls_installation_path> <destination_path>
```
where <umls_installation_path> is the path to the UMLS folder containing
the MRCONSO.RRF and MRSTY.RRF files.

You will also need to download spacy models used by QuickUMLS.
A clear message error will be displayed to show you how to install it.
Otherwise, you may also install it programmatically.

Here are examples of downloads for English and French models:
```
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
if not spacy.util.is_package("fr_core_news_sm"):
    spacy.cli.download("fr_core_news_sm")
```
:::

Given a medkit text document named `doc` with text `The patient has asthma`

```
umls_matcher = QuickUMLSMatcher(version="2021AB", language="ENG")
entities = umls_matcher.run([sentence])
```

The entity (`entities[0]`) will have the following description:
* entity.text = "asthma"
* entity.spans = [Span(16, 22)]
* entity.label = "disorder"

Its normalization attribute (`norm = entity.get_norms()[0]`) will be:
* norm is an instance of {class}`~.umls_norm_attribute.UMLSNormAttribute`
* norm.cui = _ASTHMA_CUI
* norm.umls_version = "2021AB"
* norm.term = "asthma"
* norm.score = 1.0
* norm.sem_types = ["T047"]

:::{note}
For more details about public APIs, refer to
{mod}`~.text.ner.quick_umls_matcher`.
:::

### UMLS Matcher

As an alternative to `QuickUMLSMatcher`, medkit also provides an entity matcher
dedicated to [UMLS](https://www.nlm.nih.gov/research/umls/index.html) terms
that uses the [simstring](http://chokkan.org/software/simstring/) fuzzy matching
algorithm but does not rely on `QuickUMLS`

:::{note}
For more details about public APIs, refer to {mod}`~.text.ner.umls_matcher`.
:::

### Duckling Matcher

medkit provides an entity annotator that uses [Duckling](https://github.com/facebook/duckling).

Refer to {class}`~.duckling_matcher.DucklingMatcher` for more details about requirements
for using this operation.

:::{note}
For more details about public APIs, refer to
{mod}`~.text.ner.duckling_matcher`.
:::

### EDS-NLP Date Matcher

The [EDS-NLP dates pipeline](https://aphp.github.io/edsnlp/latest/pipelines/misc/dates/) can be
directly using inside medkit to identify date and duration mentions in texts.

:::{important}
{class}`~.edsnlp_date_matcher.EDSNLPDateMatcher` needs additional dependencies that can be
installed with `pip install medkit-lib[edsnlp]`
:::

:::{note}
For more details about public APIs, refer to
{mod}`~.text.ner.edsnlp_date_matcher`.
:::

### Hugging Face Entity Matcher

medkit provides an entity matcher based on Hugging Face models.

:::{important}
{class}`~.hf_entity_matcher.HFEntityMatcher` needs additional dependencies that can be
installed with `pip install medkit-lib[hf-entity-matcher]`
:::

:::{note}
For more details about public APIs, refer to
{mod}`~.text.ner.hf_entity_matcher`.
:::

### UMLS Coder Normalizer

This operation is not an entity matcher per-say but a normalizer that will
add normalization attributes to pre-existing entities.

:::{important}
{class}`~.umls_coder_normalizer.UMLSCoderNormalizer` needs additional dependencies that can
be installed with `pip install medkit-lib[umls-coder-normalizer]`
:::

:::{note}
For more details about public APIs, refer to
{mod}`~.text.ner.umls_coder_normalizer`.
:::
### UMLS Normalization

This modules provides a subclass of
{class}`~.core.text.entity_norm_attribute.EntityNormAttribute` to facilitate
the handling of UMLS information.

:::{note}
For more details, refer to {mod}`~umls_norm_attribute`.
:::

### NLStruct Entity Matcher

Medkit provides an entity matcher for pretrained nlstruct models, these models can detect nested entities.

:::{important}
{class}`~.nlstruct_entity_matcher.NLStructEntityMatcher` needs additional dependencies that can
be installed with `pip install medkit-lib[nlstruct]`
:::

You can load directly a model from a local dir or from the HuggingFace hub.

```python
from medkit.core.text import Segment, Span
from medkit.text.ner.nlstruct_entity_matcher import NLStructEntityMatcher

text="Je lui prescris du lorazepam."
segment = Segment(text=text,spans=[Span(0,len(text))],label="test")

# define the matcher using a french model
entity_matcher = NLStructEntityMatcher(model_name_or_dirpath="NesrineBannour/CAS-privacy-preserving-model")
entities = entity_matcher.run([segment])
```

(api:text:spacy)=
## Spacy modules

medkit provides operations and utilities for wrapping spacy pipelines into
medkit. They are part of
{mod}`medkit.text.spacy` package.

:::{important}
For using this python module, you need to install [spacy](https://spacy.io/).
These dependencies may be installed with `pip install medkit-lib[spacy]`
:::

## Spacy pipelines

The {class}`~.text.spacy.SpacyPipeline` component is an annotation-level
operation. It takes medkit segments as inputs, runs a spacy pipeline, and
returns medkit segments by converting spacy outputs.

The {class}`~.text.spacy.SpacyDocPipeline` component is a document-level
operation, similarly to {class}`~.core.DocPipeline`.
It takes medkit documents as inputs, runs a spacy pipeline, and
directly attach the spacy annotations to medkit document.

:::{note}
For more info about displacy helpers, refer to {mod}`~.text.spacy.displacy_utils`.
:::

## Translation operations

:::{note}
For translation api, refer to {mod}`~.text.translation`.
:::

### HuggingFace Translator

:::{important}
{class}`~.translation.hf_translator.HFTranslator` needs additional dependencies that can
be installed with `pip install medkit-lib[hf-translator]`
:::

## Extraction of syntactic relations
This module detects syntactic relations between entities using a parser of
dependencies.

:::{note}
For more info about this module, refer to {mod}`~.text.relations.syntactic_relation_extractor`.
:::

## Post-processing modules

Medkit provides some modules to facilitate post-processing operations. 

For the moment, you can use this module to:
- align source and target {class}`~.core.text.Segment`s from the same {class}`~.core.text.TextDocument`
- duplicate attributes between segments. For example, you can duplicate an attribute from a sentence to its entities.

- filter overlapping entities: useful when creating named entity reconigtion (NER) datasets
- create mini-documents from a {class}`~.core.text.TextDocument`. 
  

```{admonition} Examples
Creating mini-documents from sections: [document splitter](../examples/text_segmentation/document.md)
```

:::{note}
For more details about public API, refer to {mod}`~.text.postprocessing`.
:::

# Metrics

This module provides components to evaluate annotations as well as some implementations of {class}`~.training.utils.MetricsComputer` to monitor the training of components in medkit. 

The components inside metrics are also known as **evaluators**. An evaluator allows you to assess performance by task.

:::{note}
For more details about public APIs, refer to {mod}`~.text.metrics`
:::

## Text Classification Evaluation

Medkit provides {class}`~.metrics.classification.TextClassificationEvaluator`, an evaluator for document attributes. You can compute the following metrics depending on your use-case:

### Classification report
-  `compute_classification_report`: To compare a list of reference and predicted documents. This method uses [sklearn](https://scikit-learn.org/stable/index.html) as backend to compute precision, recall, and F1-score.

### Inter-rated agreement
-  `compute_cohen_kappa`: To compare the degree of agreement between lists of documents made by two annotators.

-  `compute_krippendorff_alpha`: To compare the degree of agreement between lists of documents made by multiple annotators.

:::{note}
For more details about public API, refer to {class}`~.metrics.classification.TextClassificationEvaluator` or {mod}`~.text.metrics.irr_utils`.
:::

## NER Evaluation

Medkit uses [seqeval](https://github.com/chakki-works/seqeval) as backend of evaluation.

:::{important} 
This module needs additional dependencies that can be installed with `pip install medkit-lib[metrics-ner]` 
:::

### Entity detection

An example with perfect match:
- The document has two entities: PER and GPE.
- An operation has detected both entities

```
from medkit.core.text import TextDocument, Entity, Span
from medkit.text.metrics.ner import SeqEvalEvaluator

document = TextDocument("Marie lives in Paris", 
                        anns = [Entity(label="PER",spans=[Span(0,5)],text="Marie"),
                                Entity(label="GPE",spans=[Span(15,20)],text="Paris")])

pred_ents = [Entity(label="PER",spans=[Span(0,5)],text="Marie"),
             Entity(label="GPE",spans=[Span(15,20)],text="Paris")]

# define a evaluator using `iob2` as tagging scheme
evaluator = SeqEvalEvaluator(tagging_scheme="iob2")
metrics = evaluator.compute(documents=[document], predicted_entities=[pred_ents])
assert metrics["macro_precision"] == 1.0
print(metrics)
```
```
{'macro_precision': 1.0, 'macro_recall': 1.0, 'macro_f1-score': 1.0, 'support': 2, 'accuracy': 1.0, 'GPE_precision': 1.0, 'GPE_recall': 1.0, 'GPE_f1-score': 1.0, 'GPE_support': 1, 'PER_precision': 1.0, 'PER_recall': 1.0, 'PER_f1-score': 1.0, 'PER_support': 1}
```
:::{note}
For more details about public APIs, refer to {class}`~.text.metrics.ner.SeqEvalEvaluator`
:::

### Using for training of NER components

For example, a trainable component detects PER and GPE entities using `iob2` as tagging scheme. The {class}`~.training.Trainer` may compute metrics during its training/evaluation loop.

```
from medkit.text.metrics.ner import SeqEvalMetricsComputer
from medkit.training import Trainer

seqeval_mc = SeqEvalMetricsComputer(
    id_to_label={'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-GPE': 3, 'I-GPE': 4},
    tagging_scheme="iob2"
)

trainer = Trainer(
    ...
    metrics_computer=seqeval_mc
    ...
)
```

:::{note}
For more details about public APIs, refer to {class}`~.text.metrics.ner.SeqEvalMetricsComputer`. About training, refer to [training api](../api/training.md)
:::

:::{hint}
There is an utility to convert labels to NER tags if required, {mod}`~.text.ner.hf_tokenization_utils`. 
:::

:::{seealso}
You may refer to this [tutorial](../examples/finetuning_hf_model.md) to see how this works in a fine-tuning example.
:::
