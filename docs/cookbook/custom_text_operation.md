# Creating a custom text operation

If you want to initialize a custom text operation from a simple user-defined function, you can take a look to the following examples.

:::{note}
For more details about public APIs, refer to
{func}`~.text.create_text_operation`.
:::

## Filtering annotations

In this example, Jane wants to detect some entities (problems) from a raw text.

### 1. Create medkit document

```{code} python
from medkit.core.text import TextDocument

text = "The patient has asthma and is using ventoline. The patient has diabetes"
doc = TextDocument(text=text)
```

### 2. Init medkit operations

Jane would like to reuse a collegue's file containing a list of regular expression rules for detecting entities.
To this purpose, she had to split text into sentences before using the `RegexpMatcher` component.

```{code} python
from medkit.text.segmentation import SentenceTokenizer

sentence_tokenizer = SentenceTokenizer()
```

In real life, Jane should load the rules from a path using this instruction:

```{code} python
regexp_rules = RegexpMatcher.load_rules(path_to_rules_file)
```

But for this example, it is simpler for us to define this set of rules manually.

```{code} python
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule

regexp_rules = [
       RegexpMatcherRule(regexp=r"\basthma\b", label="problem"),
       RegexpMatcherRule(regexp=r"\bventoline\b", label="treatment"),
       RegexpMatcherRule(regexp=r"\bdiabetes\b", label="problem")
       ]
```

```{code} python
regexp_matcher = RegexpMatcher(rules=regexp_rules)
```

### 3. Define filter operation

As `RegexpMatcher` is based on her collegue's file, Jane would like to add a filter operation so that only entities which are problems will be returned.

For that, she has to define her own filter function and use medkit tools to instantiate this custom operation.

```{code} python
from medkit.core.text import Entity

def keep_entities_with_label_problem(entity):
    return entity.label == "problem"

from medkit.core.text import CustomTextOpType, create_text_operation

filter_operation = create_text_operation(function=keep_entities_with_label_problem, function_type=CustomTextOpType.FILTER)

# Same behavior as
# filter_operation = create_text_operation(
#   name="keep_entities_with_label_problem",
#   function=keep_entities_with_label_problem,
#   function_type=CustomTextOpType.FILTER)
```

### 4. Construct and run the pipeline

```{code} python
from medkit.core import Pipeline, PipelineStep

steps=[
    PipelineStep(input_keys=["raw_text"], output_keys=["sentences"], operation=sentence_tokenizer),
    PipelineStep(input_keys=["sentences"], output_keys=["entities"], operation=regexp_matcher),
    PipelineStep(input_keys=["entities"], output_keys=["problems"], operation=filter_operation)
]

pipeline = Pipeline(
       steps=steps,
       input_keys=["raw_text"],
       output_keys=["problems"]
)

entities = pipeline.run([doc.raw_segment])

for entity in entities:
    print(entity)
```

In this scenario, 2 entities with `problem` label are returned.

To compare with the intermediate results generated by regexpmatcher, we'll use the `entities` intermediate key.
There are 3 results.

**IMPORTANT: the following code is only for demo purpose, all pipeline steps are executed, we just select what pipeline outputs**

```{code} python
pipeline = Pipeline(
    steps=steps,
    input_keys=["raw_text"],
    output_keys=["entities"]
)

entities = pipeline.run([doc.raw_segment])

for entity in entities:
    print(entity)
```

## Creating new annotations

+++

In this example, Jane wants to pre-process the text before detecting entities.

### 1. Create medkit document

```{code} python
from medkit.core.text import TextDocument

text = """IRM : Lésion de la CPMI périphérique,
aspect distendu du LCA, kyste poplité."""

doc = TextDocument(text=text)
```

### 2. Define custom function

Jane wants to use a dictionary to convert all abbreviations into their long text.
To make it, she may define a custom function and use medkit `span_utils` to preserve spans during text modifications.

```{code} python
import re
from typing import Dict
from medkit.core.text import Segment, span_utils


# Providing the dictionary of abbreviation mapping
abbrv_mapping = {
    "IRM" : "Imagerie par Résonance Magnétique",
    "CPMI" : "Corne Postérieure du Ménisque Interne",
    "LCA" : "Ligament Croisé Antérieur",
}

# Defining custom function
def translate_abbreviations(segment, abbrv_mapping):
    ranges = []
    replacement_texts = []

    regexp = '|'.join('%s' % abbrv for abbrv in abbrv_mapping.keys())

    # Detect abbreviations
    for mo in re.finditer(regexp, segment.text):
        ranges.append([mo.start(), mo.end()])
        replacement_texts.append(abbrv_mapping[mo.group()])

    # Replace abbreviations by their text (and preserving spans)
    text, spans = span_utils.replace(
        text=segment.text,
        spans=segment.spans,
        ranges=ranges,
        replacement_texts=replacement_texts
    )

    return Segment(label="long_text", text=text, spans=spans)


from medkit.core.text import CustomTextOpType, create_text_operation

# Create the medkit operation from our custom function
preprocessing_operation = create_text_operation(
    function=translate_abbreviations,
    function_type=CustomTextOpType.CREATE_ONE_TO_N,
    name="translate_abbreviations",
    args={"abbrv_mapping":abbrv_mapping}
)
```

### 3. Run the operation

After executing the operation on the document raw text, we can observe that the output segment is composed of:
* a text with abbreviations replaced by their long text,
* spans which is a mix of modified spans (for replaced parts of text) and original spans (for not replaced text).

```{code} python
segments = preprocessing_operation.run([doc.raw_segment])

for segment in segments:
    print(f"Text: {segment.text}\n")
    print(f"Spans:")
    for span in segment.spans:
        print(f"- {span}")
```

## Extracting annotations

In this example, Jane wants to count detected UMLS cui on a set of documents.

### 1. Loading text documents

In this example, we use translated .uid documents.
For more info, you may refer to {mod}`medkit.tools.mtsamples`.

```{code} python
from medkit.tools.mtsamples import load_mtsamples

docs = load_mtsamples(nb_max=10)

print(docs[0].text)
```

### 2. Init our operations

+++

Let's initialize same operations as above (i.e., sentence tokenizer, then regexp matcher with default rules) without the filter operation.

```{code} python
from medkit.text.segmentation import SentenceTokenizer

sentence_tokenizer = SentenceTokenizer()
```

```{code} python
from medkit.text.ner import RegexpMatcher

regexp_matcher = RegexpMatcher()
```

### 3. Defining an extraction function

+++

The extraction function is defined with a label parameter for filtering entities.
Our custom operation allows to retrieve only attributes from entity with `disorder` label.

```{code} python
import re
from typing import List
from medkit.core.text import Entity, UMLSNormAttribute

# Defining custom function for extracting umls normalization attributes from entity
def extract_umls_attributes_from_entity(entity, label):
    return [attr for attr in entity.attrs.get_norms() if entity.label == label and isinstance(attr, UMLSNormAttribute) ]


from medkit.core.text import CustomTextOpType, create_text_operation

attr_extraction_operation = create_text_operation(
    function=extract_umls_attributes_from_entity,
    function_type=CustomTextOpType.EXTRACT_ONE_TO_N,
    args={"label":'disorder'}
)
```

### 4. Defining and running our pipeline

When running the pipeline on the set of documents, the output is a list of umls normalization attributes.

```{code} python
from medkit.core import Pipeline, PipelineStep

steps=[
    PipelineStep(input_keys=["raw_text"], output_keys=["sentences"], operation=sentence_tokenizer),
    PipelineStep(input_keys=["sentences"], output_keys=["entities"], operation=regexp_matcher),
    PipelineStep(input_keys=["entities"], output_keys=["umls_attributes"], operation=attr_extraction_operation),
]

pipeline = Pipeline(
       steps=steps,
       input_keys=["raw_text"],
       output_keys=["umls_attributes"] 
)
```

```{code} python
attrs = pipeline.run([doc.raw_segment for doc in docs])
attrs[:5]
```

### 5. Analyzing data

Now, Jane can analyze the number of cuis detected on her set of documents.

```{code} python
import pandas as pd
df = pd.DataFrame.from_records([attr.to_dict() for attr in attrs], columns=["cui", "umls_version"])
print(df)
```

```{code} python
df.value_counts(subset="cui")
```