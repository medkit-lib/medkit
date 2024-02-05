---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# Document splitter

+++

This tutorial will show an example of how to split a document using its sections as a reference. 

```{seealso}
We combine some operations like **section tokenizer**, **regexp matcher** and **custom operation**. Please see the other examples for more information.
```
+++

## Adding annotations in a document

Let's detect the sections and add some annotations using medkit operations.

```{code-cell} ipython3
# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/text/1.txt

from pathlib import Path
from medkit.core.text import TextDocument

doc = TextDocument.from_file(Path("../../data/text/1.txt"))
print(doc.text)
```
**Defining the operations**

```{code-cell} ipython3
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule
from medkit.text.segmentation import SectionTokenizer

# Define a section tokenizer
# The section tokenizer uses a dictionary with keywords to identify sections
section_dict = {
    "patient": ["SUBJECTIF"],
    "traitement": ["MÉDICAMENTS", "PLAN"],
    "allergies": ["ALLERGIES"],
    "examen clinique": ["EXAMEN PHYSIQUE"],
    "diagnostique": ["EVALUATION"],
}
section_tokenizer = SectionTokenizer(section_dict=section_dict)

# Define a NER operation to create 'problem', and 'treatment' entities
regexp_rules = [
    RegexpMatcherRule(regexp=r"\ballergies\b", label="problem"),
    RegexpMatcherRule(regexp=r"\basthme\b", label="problem"),
    RegexpMatcherRule(regexp=r"\ballegra\b", label="treatment", case_sensitive=False),
    RegexpMatcherRule(regexp=r"\bvaporisateurs\b", label="treatment"),
    RegexpMatcherRule(regexp=r"\bloratadine\b", label="treatment", case_sensitive=False),
    RegexpMatcherRule(regexp=r"\bnasonex\b", label="treatment", case_sensitive=False),
]
regexp_matcher = RegexpMatcher(rules=regexp_rules)
```

We can now annotate the document

```{code-cell} ipython3
# Detect annotations
sections = section_tokenizer.run([doc.raw_segment])
entities = regexp_matcher.run([doc.raw_segment])
# Annotate
for ann in sections + entities:
    doc.anns.add(ann)

print(f"The document contains {len(sections)} sections and {len(entities)} entities\n")
```

## Split the document by sections 

Once annotated, we can use the medkit operation {class}`~medkit.text.postprocessing.DocumentSplitter` to create smaller versions of the document using the sections. 

By default, since its `entity_labels`, `attr_labels`, and `relation_labels` are set to `None`, all annotations will be in the resulting documents. You can select the annotations using their labels.

```{code-cell} ipython3
from medkit.text.postprocessing import DocumentSplitter

doc_splitter = DocumentSplitter(segment_label="section", # segments of reference
                                entity_labels=["treatment","problem"],# entities to include 
                                attr_labels=[], # without attrs
                                relation_labels=[], #without relations
)
new_docs = doc_splitter.run([doc])
print(f"The document was divided into {len(new_docs)} documents\n")
```

Each document contains entities and attributes from the source segment; below, we visualize the new documents via displacy utils.

```{code-cell} ipython3
from spacy import displacy
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy

options_displacy = dict(colors={'treatment': "#85C1E9", "problem": "#ff6961"})

for new_doc in new_docs:
    print(f"New document from the section called '{new_doc.metadata['name']}'")
    # convert new document to displacy 
    displacy_data = medkit_doc_to_displacy(new_doc)
    displacy.render(displacy_data, manual=True, style="ent", options=options_displacy)
```

