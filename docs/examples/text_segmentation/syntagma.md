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

# Syntagma Tokenizer

+++

This tutorial will show an example of how to apply syntagma tokenizer medkit operation on a text document.

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

## Defining syntagma definition rules

To split the text document into medkit segments corresponding to a text part, we have to define a set of rules. 
These rules allow the operation to split the text based on regular expressions rules.

```{code-cell} ipython3
from medkit.text.segmentation.syntagma_tokenizer import SyntagmaTokenizer

separators = (
    "(?<=\. )[\w\d]+",     # Trigger: starts after a dot and space
    "(?<=\n)[\w\d]+",      # Trigger: starts after a newline
    "(?<=: )\w+",          # Trigger: starts after :
    "(?<= )mais\s+(?=\w)", # Trigger: starts with 'mais' if space before and after
    "(?<= )sans\s+(?=\w)", # Trigger: starts with 'sans' if space before and after
    "(?<= )donc\s+(?=\w)", # Trigger: starts with 'donc' if space before and after
)

tokenizer = SyntagmaTokenizer(separators)
```

The syntagmas definition is a list of regular expressions allowing to trigger the start of a new syntagma.

+++

As all operations, `SyntagmaTokenizer` defines a `run()` method. This method returns a list of {class}`~medkit.core.text.Segment` objects (a `Segment` is a
`TextAnnotation` that represents a portion of a document's full raw text). 
As input, it also expects a list of `Segment` objects. Here, we can pass a special segment containing the whole raw text of the document, that we can retrieve through the `raw_segment` attribute of `TextDocument`:

```{code-cell} ipython3
syntagmas = tokenizer.run([doc.raw_segment])

print(f"Number of detected syntagmas: {len(syntagmas)}")
print(f"Syntagmas label: {syntagmas[0].label}\n")

for syntagma in syntagmas:
    print(f"{syntagma.spans}\t{syntagma.text!r}")
```

As you can see, the text have been split into 39 medkit segments which default label is `"SYNTAGMA"`. Corresponding `spans` reflect the position of the text in the document's full raw text.

+++

## Using a yaml definition file

We have seen how to write rules programmatically.

However, it is also possible to load a yaml file containing all your rules.

First, let's create the yaml file based on previous steps.

```{code-cell} ipython3
import pathlib

filepath = pathlib.Path("syntagma.yml")

SyntagmaTokenizer.save_syntagma_definition(
    syntagma_seps=separators,
    filepath=filepath,
    encoding='utf-8')

with open(filepath, 'r') as f:
    print(f.read())
```

Now, we will see how to initialize the `SyntagmaTokenizer` operation for using this yaml file.

```{code-cell} ipython3
# Use tokenizer initialized using a yaml file
from medkit.text.segmentation import SyntagmaTokenizer

separators = SyntagmaTokenizer.load_syntagma_definition(filepath)

print("separators = ")
for sep in separators:
    print(f"- {sep!r}")

      
tokenizer = SyntagmaTokenizer(separators=separators)
```

Now let's run the operation. We can observe that the results are the same.

```{code-cell} ipython3
syntagmas = tokenizer.run([doc.raw_segment])

print(f"Number of detected syntagmas: {len(syntagmas)}\n")

for syntagma in syntagmas:
    print(f"{syntagma.spans}\t{syntagma.text!r}")
```

```{code-cell} ipython3
:tags: [remove-cell]
filepath.unlink()
```
