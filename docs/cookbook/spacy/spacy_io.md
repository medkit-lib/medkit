# Conversions to and from `spaCy`

`medkit` can load `spaCy` documents with **entities**, **attributes** (custom extensions) and groups of **spans**,
and convert documents back to `spaCy` easily.

In this example, we will show how to import `spaCy` documents into `medkit`
and how to convert `medkit` documents into `spaCy` documents.

We use some `spaCy` concepts, more information can be found in the official spacy documentation.

::::{note}
For this example, you should download the French `spaCy` model.

You can download it using:

:::{code}
import spacy.cli
spacy.cli.download("fr_core_news_sm")
:::

::::

Consider the following `spaCy` document:

:::{code}
import spacy
from spacy.tokens import Span as SpacySpan

# Load French tokenizer, tagger, parser and NER
nlp = spacy.load("fr_core_news_sm")

# Create a spacy document 
text = """Parcours patient:
Marie habite à Brest. Elle a été transférée."""
spacy_doc = nlp(text)

#  Spacy adds entities, here we add a span 'SECTION' as an example
spacy_doc.spans["SECTION"] = [SpacySpan(spacy_doc, 0, 2, "header")]

# Adding a custom attribute
# We need to define the extension before setting its value on an entity. 
# Let's define an attribute called 'country'
if not SpacySpan.has_extension("country"):
  SpacySpan.set_extension("country", default=None)

# Now, we can set the country in the 'LOC' entity
for e in spacy_doc.ents:
  if e.label_ == 'LOC':
    e._.set("country", 'France')
:::

Description of the `spaCy` document:

- Entities

:::{code}
from spacy import displacy

displacy.render(spacy_doc, style="ent")
:::

- Spans

:::{code}
displacy.render(spacy_doc, style="span", options={"spans_key": "SECTION"})
:::

The spacy document has **2** entities and **1** span group called `SECTION`.
The entity 'LOC' has **1** attribute called `country`.

Let's see how to convert this spacy doc in a `TextDocument` with annotations.

## Load a `spaCy` Doc into a list of TextDocuments

The class {class}`~medkit.io.spacy.SpacyInputConverter` is in charge of converting
`spaCy` Docs into a list of TextDocuments.

By default, it loads **all** entities, span groups and extension attributes for each SpacyDoc object,
but you can use the `entities`, `span_groups` and `attrs` parameters to specify which items should be converted,
based on their labels.

:::{tip}
You can enable provenance tracing by assigning a {class}`~medkit.core.ProvTracer` object
to the SpacyInputConverter with the `set_prov_tracer` method.
:::

:::{note}
**Span groups in medkit**

In `spaCy`, the spans are grouped with a _key_ and each span can have its own label.
To remain compatible, `medkit` uses the key as the span _label_
and the spacy label is stored as _name_ in its metadata.
:::

:::{code}
from medkit.io.spacy import SpacyInputConverter

# Define default Input Converter 
spacy_input_converter = SpacyInputConverter()

# Load spacy doc into a list of documents
docs = spacy_input_converter.load([spacy_doc])
medkit_doc = docs[0]
:::

**Description of the resulting Text document**

:::{code}
print(f"The medkit doc has {len(medkit_doc.anns)} annotations.")
print(f"The medkit doc has {len(medkit_doc.anns.get_entities())} entities.")
print(f"The medkit doc has {len(medkit_doc.anns.get_segments())} segment.")
:::

**What about 'LOC' entity?**

:::{code}
entity = medkit_doc.anns.get(label="LOC")[0]
attributes = entity.attrs.get(label="country")
print(f"Entity label={entity.label}, Entity text={entity.text}")
print("Attributes loaded from spacy")
print(attributes)
:::

**Visualizing Medkit annotations**

As explained in other tutorials, we can display `medkit` entities using `displacy`,
a visualizer developed by `spaCy`.

You can use the {func}`~medkit.text.spacy.displacy_utils.medkit_doc_to_displacy` function to format `medkit` entities.

:::{code}
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy

# getting entities in displacy format (default config) 
entities_data = medkit_doc_to_displacy(medkit_doc)
displacy.render(entities_data, style="ent",manual=True)
:::

## Convert TextDocuments to a `spaCy` Doc

Likewise, it is possible to convert a list of TextDocument to `spaCy`
using {class}`~medkit.io.spacy.SpacyOutputConverter`. 

You will need to provide a `nlp` object that tokenizes and generates
the document with the raw text as reference. By default, it converts
**all** `medkit` annotations and attributes to `spaCy`, but you can use
`anns_labels` and `attrs` parameters to specify which items should be converted. 

:::{code}
from medkit.io.spacy import SpacyOutputConverter

# define Output Converter with default params
spacy_output_converter = SpacyOutputConverter(nlp=nlp)

# Convert a list of TextDocument 

spacy_docs = spacy_output_converter.convert([medkit_doc])
spacy_doc = spacy_docs[0]

# Explore new spacy doc
print("Text of spacy doc from TextDocument:\n",spacy_doc.text)
:::

**Description of the resulting Spacy document**

- Entities imported from `medkit`

:::{code}
displacy.render(spacy_doc, style="ent")
:::

- Spans imported from `medkit`

:::{code}
displacy.render(spacy_doc, style="span",options={"spans_key": "SECTION"})
:::

**What about 'LOC' entity?**

:::{code}
entity = [e for e in spacy_doc.ents if e.label_ == 'LOC'][0]
attribute = entity._.get('country')
print(f"Entity label={entity.label_}. Entity text={entity.text}")
print("Attribute imported from medkit")
print(f"The attr `country` was imported? : {attribute is not None}, value={entity._.get('country')}")
:::
