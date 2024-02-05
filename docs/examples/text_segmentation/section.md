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


# Section Tokenizer

+++

This tutorial will show an example of how to apply section tokenizer medkit operation on a text document.

+++

## Loading a text document

For beginners, let's load a text file using the {class}`~medkit.core.text.TextDocument` class:


```{code-cell} ipython3
# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/data/text/1.txt

from pathlib import Path
from medkit.core.text import TextDocument

doc = TextDocument.from_file(Path("../../data/text/1.txt"))
```

The full raw text can be accessed through the `text` attribute:


```{code-cell} ipython3
print(doc.text)
```

## Defining section definition rules

To split the text document into medkit segments corresponding to each section, we have to define a set of rules. 
These rules allow the operation to detect keywords triggering a new section.


```{code-cell} ipython3
from medkit.text.segmentation.section_tokenizer import SectionTokenizer

section_dict = {
    "patient": ["SUBJECTIF"],
    "traitement": ["MÉDICAMENTS", "PLAN"],
    "allergies": ["ALLERGIES"],
    "examen clinique": ["EXAMEN PHYSIQUE"],
    "diagnostique": ["EVALUATION"],
}

tokenizer = SectionTokenizer(section_dict=section_dict)
```

The sections definition is a dictionary of key-values where *key* will be the section name and *value* is a list of keywords to detect as the start of the section.

For example, if we detect the keyword `EVALUATION` in text, a new section named `diagnostique` will begin with this keyword and will end with the next detected section or otherwise, the end of the text.

As all operations, `SectionTokenizer` defines a `run()` method. This method returns a list of {class}`~medkit.core.text.Segment` objects (a `Segment` is a
`TextAnnotation` that represents a portion of a document's full raw text). 
As input, it also expects a list of `Segment` objects. Here, we can pass a special segment containing the whole raw text of the document, that we can retrieve through the `raw_segment` attribute of `TextDocument`:


```{code-cell} ipython3
sections = tokenizer.run([doc.raw_segment])

print(f"Number of detected sections: {len(sections)}\n")

for section in sections:
    print(f"metadata = {section.metadata}")
    print(f"label = {section.label}")
    print(f"spans = {section.spans}")
    print(f"text = {section.text!r}\n")
```

As you can see, we have detected 6 different sections.
Each section is a segment which has:
 - an `uid` attribute, which unique value is automatically generated;
 - a `text` attribute holding the text that the segment refers to;
 - a `spans` attribute reflecting the position of this text in the document's
   full raw text. Here we only have one span for each segment, but multiple
   discontinuous spans are supported;
 - and a `label`, always equal to `"SECTION"` in our case but it could be
   different for other kinds of segments or if you initialize the operation with your own output label.
 - a `metadata` attribute, which contains a dictionary with section name value.

## Defining section rules with renaming

`SectionTokenizer` also allows to define rules (i.e., `SectionModificationRule`) for renaming detected sections based on the context of the section in the text.

Let's take the same example.


```{code-cell} ipython3
from medkit.text.segmentation.section_tokenizer import SectionTokenizer, SectionModificationRule

section_dict = {
    "patient": ["SUBJECTIF"],
    "traitement": ["MÉDICAMENTS", "PLAN"],
    "allergies": ["ALLERGIES"],
    "examen clinique": ["EXAMEN PHYSIQUE"],
    "diagnostique": ["EVALUATION"],
}
```

Now, let's add some rules for managing these cases:
- if `traitement` section is detected before `diagnostique` section, then we rename it into `traitement_entree`
- if `traitement` section is detected after `diagnostique` section, then we rename it into `traitement_sortie`


```{code-cell} ipython3
treatment_rules = [
    SectionModificationRule(
        section_name="traitement",
        new_section_name="traitement_entree",
        other_sections=["diagnostique"],
        order="BEFORE"),
    SectionModificationRule(
        section_name="traitement",
        new_section_name="traitement_sortie",
        other_sections=["diagnostique"],
        order="AFTER")
]

tokenizer = SectionTokenizer(section_dict=section_dict, section_rules=treatment_rules)
```

Let's run this new operation on document raw text.


```{code-cell} ipython3
sections = tokenizer.run([doc.raw_segment])

print(f"Number of detected sections: {len(sections)}\n")

for section in sections:
    print(f"metadata = {section.metadata}")
    print(f"label = {section.label}")
    print(f"spans = {section.spans}")
    print(f"text = {section.text!r}\n")
```

As you can see, we still detect 6 sections but 2 have been renamed into `traitement_entree` and `traitement_sortie`.

## Using a yaml definition file

We have seen how to write rules programmatically. 

However, it is also possible to load a yaml file containing all your rules.

First, let's create the yaml file based on previous steps.


```{code-cell} ipython3
import pathlib

filepath = pathlib.Path("section.yml")

SectionTokenizer.save_section_definition(
    section_dict=section_dict, 
    section_rules=treatment_rules,
    filepath=filepath,
    encoding='utf-8')

with open(filepath, 'r') as f:
    print(f.read())
```

Now, we will see how to initialize the `SectionTokenizer` operation for using this yaml file.


```{code-cell} ipython3
# Use tokenizer initialized using a yaml file
from medkit.text.segmentation.section_tokenizer import SectionTokenizer

section_dict, section_rules = SectionTokenizer.load_section_definition(filepath)

print(f"section_dict = {section_dict!r}\n")
print(f"section_rules = {section_rules!r}")

tokenizer = SectionTokenizer(section_dict=section_dict, section_rules=section_rules)
```

Now, let's run the operation. We can observe that the results are the same.


```{code-cell} ipython3
sections = tokenizer.run([doc.raw_segment])

print(f"Number of detected sections: {len(sections)}\n")

for section in sections:
    print(f"metadata = {section.metadata}")
    print(f"label = {section.label}")
    print(f"spans = {section.spans}")
    print(f"text = {section.text!r}\n")
```

```{code-cell} ipython3
:tags: [remove-cell]
filepath.unlink()
```
